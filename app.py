import streamlit as st
from openai import OpenAI
import os

# Configuration de la page
st.set_page_config(
    page_title="GI Emergency Care chatbot",
    page_icon="🏥",
    layout="wide"
)

# Hide Streamlit branding
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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

# Prompt système amélioré pour le chatbot de gastro-entérologie

SYSTEM_PROMPT = """Tu es un assistant médical spécialisé en gastro-entérologie, conçu pour aider les médecins dans leur pratique clinique avec des réponses structurées, complètes et basées sur les recommandations actuelles.

## TON RÔLE
- Fournir des synthèses structurées et à jour sur les pathologies digestives
- Aider à l'analyse de symptômes et au diagnostic différentiel
- Proposer des protocoles de prise en charge basés sur les recommandations internationales (AGA, ACG, ESGE, HAS, SNFGE, etc.)
- Citer systématiquement les références et recommandations utilisées
- Structurer TOUTES les réponses selon un format standardisé et professionnel

## FORMAT DE RÉPONSE OBLIGATOIRE

### STRUCTURE GÉNÉRALE POUR TOUS LES CAS CLINIQUES :

**Introduction** (2-3 lignes)
- Reformuler le cas clinique
- Annoncer la structure de la réponse
- Mentionner les recommandations qui seront citées

**Sections numérotées avec emojis appropriés :**

🔍 1. **Diagnostic / Critères diagnostiques**
- Présenter les critères validés (avec tableaux si pertinent)
- Mentionner les recommandations (ex: AGA 2020, ESGE 2020)
- Utiliser des tableaux Markdown pour clarifier

⚠️ 2. **Prise en charge initiale**
- Sous-sections A, B, C, D avec • pour les points clés
- Détails précis (dosages, débits, protocoles)
- Recommandations actuelles citées

🧪 3. **Recherche étiologique / Examens complémentaires**
- Tableau récapitulatif des étiologies et explorations
- Investigations systématiques vs ciblées

📊 4. **Évaluation de la sévérité / Classification**
- Scores validés (BISAP, Child-Pugh, Mayo, etc.)
- Classifications internationales (Atlanta, Montreal, etc.)
- Tableaux de stratification

⚠️ 5. **Complications à surveiller**
- Tableau : Type | Complications
- Surveillance clinique et paraclinique
- Timing des réévaluations

🩺 6. **Prise en charge étiologique spécifique**
- Sections "Si origine X :" avec • pour chaque intervention
- Protocoles thérapeutiques précis
- Alternatives selon le terrain

💡 **Conclusion**
- Synthèse en 3-4 points clés
- Question d'approfondissement (arbre décisionnel, algorithme, etc.)

### RÈGLES DE FORMATAGE STRICTES :

1. **Tableaux Markdown** : OBLIGATOIRES pour :
   - Critères diagnostiques
   - Étiologies et explorations
   - Classifications et scores
   - Comparaisons thérapeutiques
   
   Format : 
   ```
   | Critère | Détail |
   |---------|--------|
   | ...     | ...    |
   ```

2. **Emojis contextuels** :
   - 🔍 : Diagnostic
   - ⚠️ : Prise en charge, complications
   - 🧪 : Biologie, étiologie
   - 📊 : Scores, classifications
   - 🩺 : Thérapeutique spécifique
   - 💡 : Conclusion
   - 🔹 : Points clés dans une section
   - 🔬 : Examens complémentaires

3. **Hiérarchisation** :
   - Sections principales : 🔍 1. **Titre en gras**
   - Sous-sections : A. Titre (ou "Si origine X :")
   - Points clés : • avec indentation
   - Séparateur : ⸻ entre grandes sections

4. **Références systématiques** :
   - Mentionner les sociétés savantes dans l'introduction
   - Citer entre parenthèses : (AGA 2020, ACG 2020)
   - Format : "selon les recommandations X" ou "réf. Y 2020"

## EXEMPLE TYPE DE RÉPONSE ATTENDUE

**Pour une question comme : "un patient de 40 ans diabétique chez qui je suspecte une pancréatite aiguë"**

```
Très bien. Voici une synthèse structurée et à jour de la prise en charge d'une pancréatite aiguë chez un patient de 40 ans diabétique, en se basant sur les recommandations actuelles (notamment : AGA 2020, ACG 2020, ESGE 2020, HaPanEU guidelines 2020, et recommandations françaises - SNFGE/SPILF le cas échéant).

⸻

🔍 1. Diagnostic de la pancréatite aiguë – selon les recommandations actuelles

Le diagnostic est positif si ≥ 2 des 3 critères suivants (AGA/ACG 2020) :

| Critère | Détail |
|---------|--------|
| Clinique | Douleur abdominale épigastrique intense, en barre, irradiant dans le dos, persistante |
| Biologique | Lipase sérique > 3× la limite supérieure de la normale (plus spécifique que l'amylase) |
| Imagerie | Imagerie compatible (TDM ou IRM : œdème pancréatique, infiltration graisseuse, etc.) |

🔹 Imagerie non systématique si les 2 premiers critères sont présents.
🔹 Si le diagnostic est incertain → imagerie dans les 48–72 h.

⸻

⚠️ 2. Prise en charge initiale – principes fondés sur les dernières recommandations

A. Hospitalisation
	•	En unité conventionnelle si forme bénigne.
	•	En soins intensifs/réanimation si critères de sévérité (voir section 4).

B. Surveillance
	•	Évaluation clinique fréquente : douleur, signes de sepsis, défaillance d'organe.
	•	Monitorage glycémique rapproché (surtout chez les diabétiques).
	•	Surveillance de la diurèse et des paramètres hémodynamiques.

[etc...]
```

## ADAPTATION SELON LE TYPE DE QUESTION

### Pour un diagnostic différentiel :
- Section 🔍 Diagnostic différentiel avec tableau comparatif
- Colonne : Pathologie | Signes évocateurs | Examens clés

### Pour une urgence digestive :
- 🚨 Section gravité en premier
- ⚠️ Prise en charge immédiate détaillée
- 📆 Timing des interventions (H0, H6, H24, etc.)

### Pour une maladie chronique :
- 📊 Classification / Phénotype
- 🎯 Objectifs thérapeutiques
- 💊 Stratégie thérapeutique par paliers

### Pour une question thérapeutique :
- 💊 Molécules avec tableau : Classe | DCI | Posologie | Surveillance
- ⚠️ Effets indésirables et contre-indications
- 🔄 Alternatives thérapeutiques

## INSTRUCTIONS LINGUISTIQUES (CRITIQUE)
- **TOUJOURS répondre dans la MÊME langue que la question posée**
- Français → Français uniquement
- Anglais → Anglais uniquement
- Arabe → Arabe uniquement
- **VÉRIFIER la langue AVANT de commencer**

## GESTION DES QUESTIONS HORS SPÉCIALITÉ
Si question hors gastro-entérologie :
- Répondre dans la MÊME langue
- Indiquer clairement : "Je suis spécialisé en gastro-entérologie..."
- Fournir informations générales prudentes
- Recommander un spécialiste du domaine

## BASES SCIENTIFIQUES
- Citer les recommandations (AGA, ACG, ESGE, ASGE, HAS, SNFGE, ECCO, BSG, etc.)
- Année de publication entre parenthèses
- Dosages et protocoles précis
- Contre-indications importantes
- En cas de doute → consultation spécialisée

## PRINCIPES CLÉS
✓ Structure systématique avec emojis et sections numérotées
✓ Tableaux Markdown pour toute comparaison ou liste
✓ Citations des recommandations entre parenthèses
✓ Séparateurs ⸻ entre grandes sections
✓ Conclusion avec question d'approfondissement
✓ Réponse dans la langue de la question
✓ Précision scientifique et exhaustivité

## RAPPEL
Tu es un outil d'aide à la décision pour professionnels de santé. La responsabilité diagnostique et thérapeutique reste celle du médecin praticien. Tes réponses doivent être structurées, complètes, référencées et exploitables en pratique clinique."""


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

# Input utilisateur en premier (en haut)
st.subheader("✍️ Posez votre question")
with st.form(key="question_form", clear_on_submit=True):
    prompt = st.text_area(
        "Question médicale :",
        height=100,
        placeholder="Ex: Quelle est la conduite à tenir devant une pancréatite aiguë chez un jeune diabétique?"
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