from flask import Flask, jsonify, request
from flask_cors import CORS
# Inner pipeline — room/window/gate detection
from turtle_test import get_wall_json
from t import detect_gates_robust, detect_windows_json
# Outer pipeline — structural classification for material analysis
from main import get_Cordinates as get_classified_walls
from material_analysis import MaterialAnalyser, StructuralElement, build_explainability_prompt
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env file

api_key = os.getenv("API_KEY")



GEMINI_API_KEY = api_key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

app = Flask(__name__)
CORS(app)

analyser = MaterialAnalyser()

# ---------------------------------------------------------------------------
# Helper: build StructuralElement list from wall data + add slabs/columns
# ---------------------------------------------------------------------------

def _build_elements(walls: list[dict]) -> list[StructuralElement]:
    elements = []

    for w in walls:
        elements.append(StructuralElement(
            element_id=   w["element_id"],
            element_type= w["element_type"],      # load_bearing_wall | partition_wall
            room_label=   "Outer Wall" if w["is_outer"] else
                          ("Spine Wall" if w["is_spine"] else "Interior Wall"),
            span_m=       w["span_m"],
            area_m2=      round(w["span_m"] * 3.0, 2),  # assume 3 m wall height
            is_outer=     w["is_outer"],
            is_spine=     w["is_spine"],
        ))

    # Add inferred elements that OpenCV doesn't detect as lines
    # Ground floor slab — estimated from bounding box of all walls
    if walls:
        all_x = [w["start"]["x"] for w in walls] + [w["end"]["x"] for w in walls]
        all_y = [w["start"]["y"] for w in walls] + [w["end"]["y"] for w in walls]
        width_m  = (max(all_x) - min(all_x)) / 41.0   # PIXELS_PER_METRE
        depth_m  = (max(all_y) - min(all_y)) / 41.0
        area_m2  = round(width_m * depth_m, 1)
        max_span = round(max(width_m, depth_m), 2)

        elements.append(StructuralElement(
            element_id="SLAB-GF",
            element_type="slab",
            room_label="Ground Floor Slab",
            span_m=max_span,
            area_m2=area_m2,
        ))

        # Corner columns (4 corners assumed for rectangular plan)
        for i, (cx, cy) in enumerate([
            (min(all_x), min(all_y)), (max(all_x), min(all_y)),
            (min(all_x), max(all_y)), (max(all_x), max(all_y)),
        ]):
            elements.append(StructuralElement(
                element_id=f"COL-{i+1}",
                element_type="column",
                room_label=f"Corner Column {i+1}",
                span_m=0.0,
                area_m2=0.09,
            ))

    return elements


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/api/data')
def get_data():
    """Original inner route — walls + windows + gates for 3D rendering."""
    try:
        raw_coords = get_wall_json('test/F3.png')
        windows    = detect_windows_json('test/F3.png')
        mask, gates_data = detect_gates_robust('test/F3.png')

        return jsonify({
            "status": "success",
            "data":    raw_coords,
            "windows": windows,
            "gates":   gates_data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/material-analysis')
def material_analysis():
    """
    Full pipeline:
      1. Parse floor plan with structural classifier (Main.py / OpenCV)
      2. Classify walls + infer slab/columns
      3. Run material tradeoff analysis
      4. Return enriched JSON for frontend panel
    """
    try:
        walls    = get_classified_walls()
        elements = _build_elements(walls)
        analyser.analyse(elements)

        result = analyser.to_dict(elements)

        # Merge wall coordinate data into analysis results
        wall_map = {w["element_id"]: w for w in walls}
        for r in result:
            wdata = wall_map.get(r["element_id"])
            if wdata:
                r["start"]     = wdata["start"]
                r["end"]       = wdata["end"]
                r["length_px"] = wdata["length_px"]

        # Attach LLM explainability prompt text
        for i, el in enumerate(elements):
            result[i]["prompt_text"] = build_explainability_prompt(el)

        # Summary stats for the UI panel
        load_bearing = [r for r in result if r["element_type"] == "load_bearing_wall"]
        partitions   = [r for r in result if r["element_type"] == "partition_wall"]

        return jsonify({
            "status":  "success",
            "summary": {
                "total_elements":     len(result),
                "load_bearing_walls": len(load_bearing),
                "partition_walls":    len(partitions),
                "slabs":              len([r for r in result if r["element_type"] == "slab"]),
                "columns":            len([r for r in result if r["element_type"] == "column"]),
            },
            "analysis": result,
            # Raw wall coords so frontend can render both
            "walls": walls,
        })

    except Exception as e:
        import traceback
        return jsonify({"status": "error", "message": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Ask AI section using Google Gemini.
    Expects: { "question": "...", "element": { ...full element dict... } }
    """
    try:
        body     = request.get_json()
        question = body.get('question', '')
        el       = body.get('element', {})

        prompt = f"""
        You are an AI Structural Engineer. Answer the user's question about a floor plan element.
        CONTEXT:
        Element ID: {el.get('element_id')}
        Type: {el.get('element_type')}
        Span: {el.get('span_m')} meters
        Current Materials: {el.get('recommendations', [{}])[0].get('material', 'None')}

        USER QUESTION: {question}

        Answer professionally in 2-3 sentences. Cite structural safety where applicable.
        """

        response = gemini_model.generate_content(prompt)
        return jsonify({"answer": response.text})

    except Exception as e:
        return jsonify({"answer": f"Gemini Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)