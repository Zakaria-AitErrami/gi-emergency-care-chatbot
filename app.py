import streamlit as st
from openai import OpenAI
import os

# Configuration de la page
st.set_page_config(
    page_title="GI Emergency Care chatbot",
    page_icon="🏥",
    layout="wide"
)

# Initialisation du client OpenAI
@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ Clé API OpenAI non configurée. Veuillez l'ajouter dans les secrets Streamlit ou variables d'environnement.")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_openai_client()

# Configuration fixe du modèle
MODEL = "gpt-4o"
TEMPERATURE = 0.3

# Prompt système spécialisé en gastro-entérologie
SYSTEM_PROMPT = """Tu es un assistant médical spécialisé en gastro-entérologie, conçu pour aider les médecins dans leur pratique clinique. 

Ton rôle est de :
- Fournir des informations médicales précises et à jour sur les pathologies digestives
- Aider à l'analyse de symptômes et au diagnostic différentiel
- Suggérer des examens complémentaires appropriés
- Proposer des approches thérapeutiques basées sur les recommandations actuelles
- Rappeler les protocoles de prise en charge des pathologies gastro-entérologiques
- Fournir des informations sur les traitements médicamenteux et leurs interactions

Domaines d'expertise :
- Pathologies œsophagiennes (RGO, œsophagite, dysplasie, cancer)
- Pathologies gastriques (gastrite, ulcère, Helicobacter pylori, cancer)
- Pathologies intestinales (MICI, SII, diverticulose, polypes, cancer colorectal)
- Pathologies hépatiques (hépatites, cirrhose, stéatose)
- Pathologies pancréatiques et biliaires
- Troubles fonctionnels digestifs
- Endoscopie digestive et ses indications

Instructions importantes :
- Réponds dans la même langue que la question posée
- Si la question est en anglais réponds en anglais
- Si la question est en francais réponds en francais
- Si on ne comprend pas la question, dites "Je ne comprends pas la question"
- Si on te demande de répondre en une langue specifique réponds avec cette langue.
- Base tes réponses sur les recommandations scientifiques actuelles
- Cite les sociétés savantes pertinentes (SNFGE, HAS, ESGE, etc.) quand approprié
- Sois précis dans les dosages et protocoles
- Mentionne toujours les contre-indications importantes
- En cas de doute, recommande une consultation spécialisée ou des examens complémentaires
- N'hésite pas à poser des questions de clarification pour mieux comprendre le cas clinique

GESTION DES QUESTIONS HORS SPÉCIALITÉ :
Si on te pose une question sur un autre domaine médical (ophtalmologie, cardiologie, dermatologie, etc.) :
- Réponds de manière générale et courtoise
- Indique clairement : "Je suis spécialisé en gastro-entérologie et mes connaissances dans ce domaine spécifique sont limitées."
- Fournis des informations générales si tu en as, mais reste prudent
- Recommande de consulter un spécialiste du domaine concerné
- Si la question a un lien indirect avec la gastro-entérologie, mentionne ce lien le cas échéant

RAPPEL IMPORTANT : Tu es un outil d'aide à la décision. La responsabilité du diagnostic et de la prescription reste celle du médecin praticien."""

# Initialisation de l'historique dans session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Logo de l'entreprise en haut de la page
# Option 1: Si vous avez un fichier logo local
# st.image("path/to/your/logo.png", width=200)

# Option 2: Si vous avez une URL du logo
# st.image("https://your-company.com/logo.png", width=200)

# Option 3: Placeholder - Remplacez par votre logo
try:
    st.image("logo1.png", width=200)  # Assurez-vous que logo.png est dans le même dossier
except:
    st.markdown("### 🏥 Votre Logo Ici")

# Interface utilisateur
st.title("🏥 GI Emergency Care")
st.markdown("**Assistant IA spécialisé pour médecins - Aide au diagnostic et à la prise en charge en gastro-entérologie**")

# Sidebar avec informations et options
with st.sidebar:
    # Affichage du logo dans la sidebar aussi (optionnel)
    try:
        st.image("logo1.png", width=150)
    except:
        st.markdown("### 🏥")
    
    st.divider()
    
    st.header("📋 Exemples de questions")
    st.markdown("""
    - Diagnostic différentiel d'une douleur abdominale épigastrique
    - Protocole de prise en charge d'une hépatite C
    - Indications de coloscopie pour un patient de 55 ans
    - Traitement d'une maladie de Crohn active
    - Interprétation d'une élévation des transaminases
    - Conduite à tenir devant une ascite
    """)
    
    st.divider()
    
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("⚠️ Cet outil est destiné aux professionnels de santé uniquement. Il ne remplace pas le jugement clinique.")
    st.caption(f"🤖 Modèle: {MODEL}")

# Input utilisateur en premier (en haut)
st.subheader("✍️ Posez votre question")
with st.form(key="question_form", clear_on_submit=True):
    prompt = st.text_area(
        "Question médicale :",
        height=100,
        placeholder="Ex: Quels sont les critères diagnostiques de la maladie de Crohn ?"
    )
    submit_button = st.form_submit_button("Envoyer 📤", use_container_width=True)

st.divider()

# Affichage de l'historique des messages (en dessous du formulaire)
if st.session_state.messages:
    st.subheader("💬 Historique de la conversation")
else:
    st.info("👋 Commencez par poser une question ci-dessus pour démarrer la conversation.")

# Variable pour suivre si on est en train de générer
is_generating = False

for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f"**🩺 Médecin :** {message['content']}")
    else:
        st.markdown(f"**🤖 Assistant :** {message['content']}")
    st.divider()

if submit_button and prompt:
    # Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Conteneur pour la nouvelle réponse qui s'affichera en bas
    st.markdown("---")
    st.markdown(f"**🩺 Médecin :** {prompt}")
    st.markdown("**🤖 Assistant :**")
    response_container = st.empty()
    
    # Génération de la réponse avec streaming en temps réel
    try:
        # Préparation des messages pour l'API
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(st.session_state.messages)
        
        # Appel à l'API OpenAI avec streaming
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            stream=True
        )
        
        # Affichage en temps réel de la réponse
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                # Mise à jour en temps réel avec un curseur
                response_container.markdown(full_response + "▌")
        
        # Affichage final sans le curseur
        response_container.markdown(full_response)
        
        # Ajout de la réponse à l'historique
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération de la réponse : {str(e)}")
        full_response = "Désolé, une erreur s'est produite. Veuillez réessayer."
        response_container.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.divider()
st.caption("💡 **Disclaimer** : Cet assistant utilise l'IA pour fournir des informations médicales. Les informations fournies ne constituent pas un avis médical définitif et doivent être validées par un professionnel de santé qualifié.")