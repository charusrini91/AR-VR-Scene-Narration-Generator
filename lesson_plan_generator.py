import streamlit as st
import textwrap
import json
from datetime import datetime
import os

# ---------- Template prompts & generators (offline) ----------
TEMPLATE_PROMPTS = {
    "overview": (
        "Create a concise lesson overview for teachers that introduces the topic '{topic}' "
        "for grade {grade}. Duration: {duration}."
    ),
}

def offline_generate(topic: str, grade: str, duration: str) -> dict:
    """Deterministic lesson plan generator (template/offline)."""
    overview = textwrap.fill(
        f"This lesson introduces {topic} to Grade {grade} students in a {duration} session. "
        "It focuses on clear objectives, active learning and assessment for understanding.",
        width=80
    )

    objectives = "\n".join([
        f"1. Students will be able to explain the key idea of {topic}.",
        "2. Students will demonstrate understanding through a short task or model.",
        "3. Students will apply the concept in a real-world or problem-solving context."
    ])

    standards = "Aligns with local standards. Tie objectives to subject standards."

    materials = "- Whiteboard/markers\n- Student notebooks\n- Printed worksheet / digital doc\n- Multimedia (video or images)"

    warmup = f"Quick 5-minute question: 'What do you already know about {topic}?' Use think-pair-share."

    main_activity = (
        "1. Hook (2-3 min): Show an image/video and ask a guiding question.\n"
        "2. Direct instruction (8-10 min): Teacher explains concept with examples.\n"
        "3. Guided practice (10-12 min): Students work in pairs on a worksheet/activity; teacher circulates.\n"
        "4. Independent application (8-10 min): Students complete a short task individually."
    )

    assessment = (
        "Formative:\n"
        " - Exit ticket: 3 quick questions that check the objectives.\n"
        " - Teacher observation checklist during group work.\n\n"
        "Summative:\n - Short quiz (5 items) or a performance task graded against a 3-point rubric."
    )

    differentiation = (
        "Supporting struggling learners:\n"
        "- Provide sentence starters and scaffolded worksheet.\n"
        "- Pair with a peer mentor; allow alternative means of expression.\n"
        "- Break tasks into smaller steps with checklists.\n\n"
        "Extensions for advanced learners:\n"
        "- Offer an open-ended project or deeper research prompt.\n"
        "- Ask them to design a mini-teaching segment for peers."
    )

    closure = "Ask each student to write one thing they learned and one question they still have. Collect exit tickets."

    homework = (
        "1) Short reflection paragraph applying the concept to a real-world example.\n"
        "2) Optional research or creative project for extra credit."
    )

    plan = {
        "title": f"{topic} — Grade {grade} Lesson Plan",
        "overview": overview,
        "objectives": objectives,
        "standards": standards,
        "materials": materials,
        "warmup": warmup,
        "main_activity": main_activity,
        "assessment": assessment,
        "differentiation": differentiation,
        "closure": closure,
        "homework": homework,
        "metadata": {
            "topic": topic,
            "grade": grade,
            "duration": duration,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "mode": "template"
        }
    }
    return plan

# ---------- Optional AI generator (uses openai package) ----------

def ai_generate_openai(topic: str, grade: str, duration: str, openai_module) -> dict:
    """Call an OpenAI-like client and request a JSON lesson plan.
    The app will ask for user's API key and set openai.api_key before calling.
    """
    system_prompt = (
        "You are a helpful lesson plan generator. Produce a JSON object with keys: "
        "\"title\",\"overview\",\"objectives\",\"standards\",\"materials\",\"warmup\",\"main_activity\",\"assessment\","
        "\"differentiation\",\"closure\",\"homework\". Keep each value as a short string or multi-line string. "
        "Make content practical and classroom-ready."
    )

    user_prompt = (
        f"Create a lesson plan for topic: {topic}\n"
        f"Grade: {grade}\n"
        f"Duration: {duration}\n"
        "Return only a valid JSON object (no extra commentary)."
    )

    # different OpenAI clients / versions vary; we try a common pattern
    try:
        resp = openai_module.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.7,
        )
        content = resp.choices[0].message.content
    except Exception:
        # fallback to older attribute names
        resp = openai_module.Completion.create(
            model="text-davinci-003",
            prompt=system_prompt + "\n" + user_prompt,
            max_tokens=800,
            temperature=0.7,
        )
        content = resp.choices[0].text

    try:
        parsed = json.loads(content)
    except Exception:
        # If the model didn't return strict JSON, wrap content into overview
        parsed = {
            "title": f"{topic} — Grade {grade} Lesson Plan (AI)",
            "overview": content,
            "objectives": "",
            "standards": "",
            "materials": "",
            "warmup": "",
            "main_activity": "",
            "assessment": "",
            "differentiation": "",
            "closure": "",
            "homework": "",
        }

    parsed.setdefault("metadata", {})
    parsed["metadata"].update({
        "topic": topic,
        "grade": grade,
        "duration": duration,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "mode": "ai"
    })
    return parsed

# ---------- Utilities ----------

def render_markdown(plan: dict) -> str:
    md = []
    md.append(f"# {plan.get('title', 'Lesson Plan')}\n")
    meta = plan.get("metadata", {})
    md.append(f"**Topic:** {meta.get('topic','')}\n\n")
    md.append(f"**Grade:** {meta.get('grade','')}\n\n")
    md.append(f"**Duration:** {meta.get('duration','')}\n\n")
    md.append("---\n")
    for section in ["overview","objectives","standards","materials","warmup","main_activity","assessment","differentiation","closure","homework"]:
        if plan.get(section):
            md.append(f"## {section.replace('_',' ').title()}\n")
            md.append(plan[section] + "\n\n")
    md.append(f"---\n*Generated at {meta.get('generated_at','unknown')} ({meta.get('mode','')})*\n")
    return "\n".join(md)

# ---------- Streamlit UI ----------

def main():
    st.set_page_config(page_title="Lesson Plan Generator", layout="centered")
    st.title("AI Lesson Plan Generator — Streamlit")
    st.markdown("Create classroom-ready lesson plans quickly. Use Template mode (offline) or AI mode (requires OpenAI key).")

    with st.form("inputs"):
        topic = st.text_input("Topic", value="Photosynthesis")
        grade = st.text_input("Grade / Level", value="6")
        duration = st.text_input("Duration", value="45 minutes")
        mode = st.selectbox("Mode", options=["template", "ai"], index=0)
        openai_key = st.text_input("OpenAI API Key (only if using AI mode)", type="password")
        out_name = st.text_input("Output filename (optional, .md will be appended if missing)", value="lesson_plan")
        submitted = st.form_submit_button("Generate Lesson Plan")

    if submitted:
        if mode == "template":
            plan = offline_generate(topic, grade, duration)
        else:
            if not openai_key:
                st.error("AI mode selected but no OpenAI API key provided. Enter your key or switch to Template mode.")
                st.stop()
            try:
                import openai
                openai.api_key = openai_key
                plan = ai_generate_openai(topic, grade, duration, openai)
            except ModuleNotFoundError:
                st.error("openai package not installed. Install it with: pip install openai")
                st.stop()
            except Exception as e:
                st.warning(f"AI generation failed: {e}. Falling back to template generator.")
                plan = offline_generate(topic, grade, duration)

        md = render_markdown(plan)

        st.subheader("Generated Lesson Plan")
        st.markdown(md)

        # Download as Markdown
        filename = out_name if out_name.endswith('.md') else out_name + '.md'
        st.download_button(label="Download as .md", data=md, file_name=filename, mime="text/markdown")

        # Save to server (useful if running locally)
        try:
            save_dir = "generated_lessons"
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, filename)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(md)
            st.success(f"Saved copy to: {path}")
        except Exception as e:
            st.warning(f"Could not save file to disk: {e}")

        # Optionally show JSON
        if st.checkbox("Show JSON"):
            st.json(plan)

if __name__ == '__main__':
    main()
