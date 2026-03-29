from flask import Flask, jsonify, request
from flask_cors import CORS
from Main import get_Cordinates
import anthropic
from material_analysis import MaterialAnalyser, StructuralElement, build_explainability_prompt
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyCo-rslFCMoyBo6P5EkAB4e1rKlOnmhEWg"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

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
    """Original route — walls only (keeps frontend working as-is)."""
    try:
        walls = get_Cordinates()
        return jsonify({"status": "success", "data": walls})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/material-analysis')
def material_analysis():
    """
    Full pipeline:
      1. Parse floor plan (OpenCV)
      2. Classify walls + infer slab/columns
      3. Run material tradeoff analysis
      4. Return enriched JSON for frontend panel
    """
    try:
        walls = get_Cordinates()
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

        # Attach the LLM explainability prompt text (frontend can display as-is,
        # or you can call an LLM API here and replace prompt_text with explanation)
        for i, el in enumerate(elements):
            result[i]["prompt_text"] = build_explainability_prompt(el)

        # Summary stats for the UI panel
        load_bearing = [r for r in result if r["element_type"] == "load_bearing_wall"]
        partitions   = [r for r in result if r["element_type"] == "partition_wall"]

        return jsonify({
            "status":   "success",
            "summary": {
                "total_elements":     len(result),
                "load_bearing_walls": len(load_bearing),
                "partition_walls":    len(partitions),
                "slabs":              len([r for r in result if r["element_type"] == "slab"]),
                "columns":            len([r for r in result if r["element_type"] == "column"]),
            },
            "analysis": result,
            # Also return raw wall coords so frontend can render both
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
    """
    try:
        body = request.get_json()
        question = body.get('question', '')
        el = body.get('element', {})

        # System Prompt for Gemini
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

        # Gemini API Call
        response = model.generate_content(prompt)
        
        return jsonify({"answer": response.text})

    except Exception as e:
        return jsonify({"answer": f"Gemini Error: {str(e)}"}), 500
# @app.route('/api/chat', methods=['POST'])
# def chat():
#     """
#     Per-element cross-questioning.
#     Expects: { "question": "...", "element": { ...full element dict... } }
#     Returns: { "answer": "..." }

#     Uses the Anthropic SDK if available, falls back to a rule-based response.
#     """
#     try:
#         body    = request.get_json()
#         question = body.get('question', '').strip()
#         el       = body.get('element', {})

#         if not question:
#             return jsonify({"answer": "Please ask a question."})

#         # Build a rich system prompt grounded in real element data
#         recs = el.get('recommendations', [])[:3]
#         rec_text = "\n".join(
#             f"  #{i+1} {r['material']} — score {r['score']:.3f}, "
#             f"Cost:{r['cost_label']}, Str:{r['strength_label']}, Dur:{r['durability_label']}, "
#             f"₹{r['unit_cost_inr']:,}/m³. {r['notes']}"
#             for i, r in enumerate(recs)
#         )
#         concerns_text = "\n".join(f"  ⚠ {c}" for c in el.get('concerns', [])) or "  None."
#         wp = el.get('weight_profile', {})

#         system = f"""You are a structural engineering assistant embedded in a 3D floor plan tool.
# The user is asking about a specific structural element. Answer concisely (2–4 sentences max).
# Cite actual numbers — span, score, cost — don't give vague answers.

# Element: {el.get('element_id')} ({el.get('element_type','').replace('_',' ')})
# Room: {el.get('room_label','N/A')}
# Span: {el.get('span_m')} m | Area: {el.get('area_m2')} m² | Outer: {el.get('is_outer')}

# Scoring weights: strength={wp.get('strength','?')}, durability={wp.get('durability','?')}, cost={wp.get('cost','?')}
# Rationale: {wp.get('description','')}

# Top material recommendations:
# {rec_text}

# Structural concerns:
# {concerns_text}
# """

#         # Try Anthropic SDK
#         try:
#             client = anthropic.Anthropic()
#             msg = client.messages.create(
#                 model="claude-sonnet-4-20250514",
#                 max_tokens=300,
#                 system=system,
#                 messages=[{"role": "user", "content": question}]
#             )
#             answer = msg.content[0].text
#         except Exception:
#             # Fallback: rule-based answer using element data
#             top = recs[0] if recs else {}
#             answer = (
#                 f"{el.get('element_id')} is a {el.get('element_type','').replace('_',' ')} "
#                 f"spanning {el.get('span_m')} m. "
#                 f"The top recommendation is {top.get('material','—')} "
#                 f"(score {top.get('score',0):.3f}) due to its {top.get('strength_label','—')} strength "
#                 f"at ₹{top.get('unit_cost_inr',0):,}/m³. "
#                 f"{top.get('notes','')}"
#             )

#         return jsonify({"answer": answer})

#     except Exception as e:
#         return jsonify({"answer": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
