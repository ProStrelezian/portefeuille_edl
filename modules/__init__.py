import streamlit as st

def stream_llm_response(prompt, user_api_key="", history=None):
    openai_key = ""
    gemini_key = ""
    
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass
        
    try:
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
        
    if user_api_key and user_api_key.startswith("sk-"):
        openai_key = user_api_key
    elif user_api_key:
        gemini_key = user_api_key
    
    if not openai_key and not gemini_key:
        yield "⚠️ Aucune clé API trouvée. Ajoutez une clé API dans le menu latéral (Optionnel) ou via un fichier `secrets.toml`."
        return

    # history contains list of dicts: [{"role": "user"/"assistant", "content": "..."}]
    
    try:
        if openai_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            messages = []
            if history:
                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=800,
                temperature=0.7,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        elif gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            gemini_history = []
            if history:
                for msg in history:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_history.append({"role": role, "parts": [msg["content"]]})
            
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
    except Exception as e:
        yield f"Erreur lors de la communication avec l'IA: {e}"
