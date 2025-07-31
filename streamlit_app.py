# streamlit_app.py
"""
TalentScout - AI Hiring Assistant
---------------------------------
A sophisticated hiring assistant chatbot that collects candidate information,
generates technical questions, and provides an enhanced user experience with
multilingual support, sentiment analysis, and robust conversation handling.

Features:
- Comprehensive candidate information gathering
- Tech stack-based technical question generation
- Multilingual support (bonus feature)
- Sentiment analysis (bonus feature)
- Robust fallback mechanisms
- Data privacy and security compliance
- Context-aware conversation flow
"""

import streamlit as st
import openai
import os
import re
import hashlib
import json
from dotenv import load_dotenv
from typing import Dict, List, Optional
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize OpenAI client with error handling
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), 
        base_url=os.getenv("OPENAI_BASE_URL")
    )
except Exception as e:
    st.error("⚠️ OpenAI configuration error. Please check your API key and base URL.")
    st.stop()

# Debug function to test API connectivity
def test_api_connection():
    """Test OpenAI API connection and available models."""
    try:
        # Simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        return True, "API connection successful"
    except Exception as e:
        return False, str(e)

# Configuration and Constants
LANGUAGES = {
    "English": "en",
    "Español": "es", 
    "Français": "fr",
    "Deutsch": "de",
    "हिन्दी": "hi",
    "中文": "zh",
}

EXIT_KEYWORDS = [
    "exit", "quit", "bye", "goodbye", "end", "stop", "terminate",
    "finish", "done", "cancel", "leave", "close"
]

# Core purpose definition for fallback mechanism
CORE_PURPOSE = """
You are TalentScout's AI Hiring Assistant. Your ONLY purpose is to:
1. Collect candidate information for recruitment
2. Generate technical interview questions based on tech stack
3. Assist with the hiring screening process

You MUST NOT:
- Provide general advice unrelated to hiring
- Answer non-recruitment questions
- Engage in casual conversation beyond professional courtesy
- Provide information outside your hiring assistant role
"""

# UI Configuration and Styling 
st.set_page_config(
    page_title="TalentScout Hiring Assistant", 
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {background-color: #f7f9fa; padding: 1rem;}
    .stButton>button {
        background-color: #4F8BF9; 
        color: white; 
        font-weight: bold; 
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3a7bd5;
        transform: translateY(-1px);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #4F8BF9;
    }
    .candidate-info {
        background-color: #e8f2ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4F8BF9;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

#  Security and Data Privacy Functions 
def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks and ensure data privacy.
    """
    if not text:
        return ""
    
    # Remove potentially harmful characters
    sanitized = re.sub(r'[<>"\';()&]', '', text)
    # Limit length to prevent abuse
    sanitized = sanitized[:500]
    return sanitized.strip()

def hash_sensitive_data(data: str) -> str:
    """
    Hash sensitive data for privacy compliance (GDPR).
    """
    return hashlib.sha256(data.encode()).hexdigest()[:16]

def validate_email(email: str) -> bool:
    """
    Validate email format.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    """
    Validate phone number format (basic validation).
    """
    pattern = r'^[\+]?[1-9][\d\s\-\(\)]{7,14}$'
    return re.match(pattern, phone.replace(' ', '').replace('-', '')) is not None

#  Session State Initialization
def initialize_session_state():
    """
    Initialize all session state variables with proper defaults.
    """
    defaults = {
        "stage": "greeting",
        "candidate_info": {},
        "language": "English", 
        "conversation_history": [],
        "user_preferences": {},
        "greeted": False,
        "questions_generated": False,
        "current_question_set": "",
        "fallback_count": 0,
        "session_id": hash_sensitive_data(str(datetime.now())),
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Translation and Multilingual Support

# Basic translation dictionary for fallback
BASIC_TRANSLATIONS = {
    "Spanish": {
        "Full Name": "Nombre Completo",
        "Email Address": "Dirección de Correo",
        "Phone Number": "Número de Teléfono",
        "Years of Experience": "Años de Experiencia",
        "Desired Position": "Posición Deseada",
        "Current Location": "Ubicación Actual",
        "Technical Skills & Technologies": "Habilidades Técnicas y Tecnologías",
        "Generate Questions": "Generar Preguntas",
        "Complete Application": "Completar Aplicación",
        "Thank you": "Gracias",
        "Welcome": "Bienvenido"
    },
    "French": {
        "Full Name": "Nom Complet",
        "Email Address": "Adresse E-mail",
        "Phone Number": "Numéro de Téléphone",
        "Years of Experience": "Années d'Expérience",
        "Desired Position": "Poste Souhaité",
        "Current Location": "Lieu Actuel",
        "Technical Skills & Technologies": "Compétences Techniques et Technologies",
        "Generate Questions": "Générer des Questions",
        "Complete Application": "Terminer la Candidature",
        "Thank you": "Merci",
        "Welcome": "Bienvenue"
    }
}

def simple_translate(text: str, target_lang: str) -> str:
    """Simple translation using dictionary lookup for common phrases."""
    if target_lang == "English":
        return text
    
    lang_key = target_lang.split()[0]  # Get first word (e.g., "Spanish" from "Español")
    if lang_key in BASIC_TRANSLATIONS:
        return BASIC_TRANSLATIONS[lang_key].get(text, text)
    return text

def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text to target language using OpenAI API with improved error handling.
    """
    if target_lang == "English" or not text.strip():
        return text
    
    try:
        prompt = f"""Translate the following text to {target_lang}. 
        Keep the tone professional and preserve any technical terms:
        
        {text}"""
        
        # Try multiple model options for better compatibility
        models_to_try = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception as model_error:
                continue
                
        # If all models fail, show specific error
        raise Exception("No available models for translation")
        
    except Exception as e:
        # Try simple translation first
        simple_result = simple_translate(text, target_lang)
        if simple_result != text:
            return simple_result
            
        # Better error handling with specific error info
        error_msg = str(e)
        if "rate_limit" in error_msg.lower():
            st.warning("⚠️ Translation rate limit reached. Using basic translation.")
        elif "api_key" in error_msg.lower():
            st.warning("⚠️ API key issue. Using basic translation. Check your OpenAI configuration.")
        else:
            st.warning(f"⚠️ AI translation unavailable. Using basic translation.")
        return text

#  Enhanced Prompt Engineering 
def generate_technical_questions(tech_stack: str, experience_level: int, position: str, language: str = "English") -> str:
    """
    Generate technical questions using sophisticated prompt engineering.
    """
    # Determine difficulty level based on experience
    if experience_level <= 2:
        difficulty = "beginner to intermediate"
        depth = "fundamental concepts and basic implementation"
    elif experience_level <= 5:
        difficulty = "intermediate to advanced"
        depth = "design patterns, best practices, and problem-solving"
    else:
        difficulty = "advanced to expert"
        depth = "architecture, optimization, and leadership scenarios"
    
    base_prompt = f"""
    {CORE_PURPOSE}
    
    Generate exactly 4 technical interview questions for a {position} candidate with {experience_level} years of experience.
    
    Tech Stack: {tech_stack}
    Difficulty Level: {difficulty}
    Focus Areas: {depth}
    
    Requirements:
    1. Questions should be directly relevant to the specified technologies
    2. Appropriate for {difficulty} level
    3. Mix of conceptual and practical questions
    4. Include at least one problem-solving scenario
    5. Questions should assess real-world application knowledge
    
    Format each question clearly with proper numbering.
    """
    
    if language != "English":
        base_prompt += f"\n\nProvide all questions in {language} language."
    
    try:
        with st.spinner(translate_text("Generating personalized technical questions...", language)):
            # Try multiple model options for better compatibility
            models_to_try = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
            
            for model in models_to_try:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": base_prompt}],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    return response.choices[0].message.content.strip()
                except Exception:
                    continue
            
            # If all models fail
            raise Exception("No available models for question generation")
            
    except Exception as e:
        error_msg = translate_text(
            f"Unable to generate questions at this time. Please try again later.", 
            language
        )
        return error_msg

# Fallback Mechanism 
def handle_fallback_response(user_input: str, language: str = "English") -> str:
    """
    Handle unexpected inputs with purpose-focused fallback responses.
    """
    fallback_prompt = f"""
    {CORE_PURPOSE}
    
    A user said: "{user_input}"
    
    Provide a helpful, professional response that:
    1. Politely redirects them back to the hiring process
    2. Explains what you can help with
    3. Asks a relevant follow-up question to continue the hiring conversation
    4. Maintains a friendly but professional tone
    
    Keep the response concise (2-3 sentences maximum).
    """
    
    if language != "English":
        fallback_prompt += f"\n\nRespond in {language}."
    
    try:
        # Try multiple model options for better compatibility
        models_to_try = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": fallback_prompt}],
                    temperature=0.5,
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
            except Exception:
                continue
                
        # If all models fail, use default responses
        raise Exception("No available models for fallback")
        
    except Exception:
        default_responses = {
            "English": "I'm here to assist with your job application and technical screening. Could you please provide the requested information to continue?",
            "Español": "Estoy aquí para ayudarte con tu solicitud de trabajo y evaluación técnica. ¿Podrías proporcionar la información solicitada para continuar?",
            "Français": "Je suis là pour vous aider avec votre candidature et l'évaluation technique. Pourriez-vous fournir les informations demandées pour continuer?",
        }
        return default_responses.get(language, default_responses["English"])

# Sentiment Analysis
def analyze_sentiment(text: str, language: str = "English") -> Dict[str, str]:
    """
    Enhanced sentiment analysis with confidence scoring.
    """
    if not text.strip():
        return {"sentiment": "Neutral", "confidence": "N/A", "note": "No response provided"}
    
    prompt = f"""
    Analyze the sentiment and confidence level of this candidate response:
    "{text}"
    
    Provide:
    1. Sentiment: Positive, Neutral, or Negative
    2. Confidence: High, Medium, or Low
    3. Brief note about the candidate's apparent engagement level
    
    Response format:
    Sentiment: [sentiment]
    Confidence: [confidence]
    Note: [brief note]
    """
    
    try:
        # Try multiple model options for better compatibility
        models_to_try = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200
                )
                
                result = response.choices[0].message.content.strip()
                
                # Parse the response
                sentiment_match = re.search(r'Sentiment:\s*(\w+)', result)
                confidence_match = re.search(r'Confidence:\s*(\w+)', result)
                note_match = re.search(r'Note:\s*(.+)', result)
                
                return {
                    "sentiment": sentiment_match.group(1) if sentiment_match else "Neutral",
                    "confidence": confidence_match.group(1) if confidence_match else "Medium",
                    "note": note_match.group(1) if note_match else "Standard response"
                }
            except Exception:
                continue
                
        # If all models fail
        raise Exception("No available models for sentiment analysis")
        
    except Exception:
        return {"sentiment": "Neutral", "confidence": "Unknown", "note": "Analysis unavailable"}

# Conversation Management
def check_exit_intent(user_input: str) -> bool:
    """
    Check if user wants to exit the conversation.
    """
    if not user_input:
        return False
    
    user_input_lower = user_input.lower().strip()
    return any(keyword in user_input_lower for keyword in EXIT_KEYWORDS)

def add_to_conversation_history(role: str, content: str, metadata: Optional[Dict] = None):
    """
    Add interaction to conversation history for context.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content,
        "metadata": metadata or {}
    }
    st.session_state.conversation_history.append(entry)

# Initialize Session
initialize_session_state()

# Header
st.title("🤖 TalentScout - AI Hiring Assistant")
st.markdown("*Your intelligent partner in technical recruitment*")

# Language Selection Sidebar
with st.sidebar:
    st.header("🌐 Language / Idioma")
    selected_lang = st.selectbox(
        "Choose your preferred language:",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.language)
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Session Info**")
    st.caption(f"Session ID: {st.session_state.session_id}")
    st.caption(f"Current Stage: {st.session_state.stage.title()}")
    
    # API Connection Test
    st.markdown("---")
    st.markdown("**API Status**")
    if st.button("🔧 Test API Connection"):
        with st.spinner("Testing API connection..."):
            success, message = test_api_connection()
            if success:
                st.success("✅ API connection working")
            else:
                st.error(f"❌ API Error: {message}")
                if "api_key" in message.lower():
                    st.info("💡 Check your OPENAI_API_KEY in .env file")
                elif "model" in message.lower():
                    st.info("💡 Try using gpt-3.5-turbo model")
    
    if st.button("🔄 Start New Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main Application Flow

# Stage 1: Greeting
if st.session_state.stage == "greeting":
    if not st.session_state.greeted:
        greeting_msg = translate_text(
            """Hello! 👋 Welcome to TalentScout, your AI-powered hiring assistant.

I'm here to help streamline your job application process by:
• Collecting your professional information
• Understanding your technical expertise
• Generating relevant interview questions based on your skills

This process typically takes 5-10 minutes. You can type 'exit' at any time to end our conversation.

Ready to get started?""", 
            st.session_state.language
        )
        st.info(greeting_msg)
        st.session_state.greeted = True
    
    user_response = st.text_input(
        translate_text("Type 'yes' to begin or 'exit' to quit:", st.session_state.language),
        key="greeting_input"
    )
    
    if user_response:
        user_response = sanitize_input(user_response)
        
        if check_exit_intent(user_response):
            farewell_msg = translate_text(
                "Thank you for considering TalentScout. We hope to assist you in the future. Goodbye! 👋",
                st.session_state.language
            )
            st.success(farewell_msg)
            st.session_state.stage = "ended"
            st.rerun()
        elif user_response.lower().strip() in ['yes', 'y', 'start', 'begin', 'ok', 'sure']:
            add_to_conversation_history("user", user_response)
            st.session_state.stage = "collect_basic_info"
            st.rerun()
        else:
            # Fallback mechanism
            fallback_response = handle_fallback_response(user_response, st.session_state.language)
            st.warning(fallback_response)
            st.session_state.fallback_count += 1

# Stage 2: Basic Information Collection
elif st.session_state.stage == "collect_basic_info":
    st.subheader(translate_text("📝 Basic Information", st.session_state.language))
    st.markdown(translate_text("Please provide your basic contact and professional details:", st.session_state.language))
    
    with st.form("basic_info_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input(
                translate_text("Full Name *", st.session_state.language),
                value=st.session_state.candidate_info.get("name", ""),
                help=translate_text("Enter your complete legal name", st.session_state.language)
            )
            email = st.text_input(
                translate_text("Email Address *", st.session_state.language),
                value=st.session_state.candidate_info.get("email", ""),
                help=translate_text("Professional email address preferred", st.session_state.language)
            )
            phone = st.text_input(
                translate_text("Phone Number *", st.session_state.language),
                value=st.session_state.candidate_info.get("phone", ""),
                help=translate_text("Include country code if international", st.session_state.language)
            )
        
        with col2:
            experience = st.slider(
                translate_text("Years of Experience *", st.session_state.language),
                0, 30, 
                value=st.session_state.candidate_info.get("experience", 2),
                help=translate_text("Total professional experience", st.session_state.language)
            )
            position = st.text_input(
                translate_text("Desired Position *", st.session_state.language),
                value=st.session_state.candidate_info.get("position", ""),
                help=translate_text("E.g., Software Engineer, Data Scientist", st.session_state.language)
            )
            location = st.text_input(
                translate_text("Current Location", st.session_state.language),
                value=st.session_state.candidate_info.get("location", ""),
                help=translate_text("City, Country", st.session_state.language)
            )
        
        submitted = st.form_submit_button(
            translate_text("Continue to Tech Stack →", st.session_state.language),
            use_container_width=True
        )
    
    if submitted:
        # Check for exit intent
        if any(check_exit_intent(val) for val in [name, email, phone, position, location]):
            st.success(translate_text("Thank you for your time. Goodbye!", st.session_state.language))
            st.session_state.stage = "ended"
            st.rerun()
        
        # Validate required fields
        errors = []
        if not name.strip():
            errors.append(translate_text("Full name is required", st.session_state.language))
        if not email.strip():
            errors.append(translate_text("Email address is required", st.session_state.language))
        elif not validate_email(email):
            errors.append(translate_text("Please enter a valid email address", st.session_state.language))
        if not phone.strip():
            errors.append(translate_text("Phone number is required", st.session_state.language))
        elif not validate_phone(phone):
            errors.append(translate_text("Please enter a valid phone number", st.session_state.language))
        if not position.strip():
            errors.append(translate_text("Desired position is required", st.session_state.language))
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            # Sanitize and store data
            st.session_state.candidate_info.update({
                "name": sanitize_input(name),
                "email": sanitize_input(email.lower()),
                "phone": sanitize_input(phone),
                "experience": experience,
                "position": sanitize_input(position),
                "location": sanitize_input(location) if location else "Not specified"
            })
            
            add_to_conversation_history("assistant", "Basic information collected successfully")
            st.session_state.stage = "collect_tech_stack"
            st.rerun()

# Stage 3: Tech Stack Collection
elif st.session_state.stage == "collect_tech_stack":
    st.subheader(translate_text("⚡ Technical Stack", st.session_state.language))
    
    # Show candidate summary
    with st.expander(translate_text("📋 Your Information Summary", st.session_state.language), expanded=False):
        st.markdown(f"**{translate_text('Name', st.session_state.language)}:** {st.session_state.candidate_info['name']}")
        st.markdown(f"**{translate_text('Position', st.session_state.language)}:** {st.session_state.candidate_info['position']}")
        st.markdown(f"**{translate_text('Experience', st.session_state.language)}:** {st.session_state.candidate_info['experience']} {translate_text('years', st.session_state.language)}")
    
    st.markdown(translate_text(
        "Please describe your technical expertise in detail. Include programming languages, frameworks, databases, tools, and any other relevant technologies:",
        st.session_state.language
    ))
    
    # Tech stack examples
    with st.expander(translate_text("💡 Examples", st.session_state.language)):
        examples = translate_text("""
        Frontend Developer: React, JavaScript, TypeScript, HTML5, CSS3, Redux, Webpack, Jest
        
        Backend Developer: Python, Django, PostgreSQL, Redis, Docker, AWS, REST APIs
        
        Full Stack: JavaScript, Node.js, Express, React, MongoDB, Git, Heroku
        
        Data Scientist: Python, Pandas, NumPy, Scikit-learn, TensorFlow, SQL, Jupyter, AWS
        """, st.session_state.language)
        st.markdown(examples)
    
    tech_stack = st.text_area(
        translate_text("Technical Skills & Technologies *", st.session_state.language),
        value=st.session_state.candidate_info.get("tech_stack", ""),
        height=120,
        help=translate_text("Be specific about your experience level with each technology", st.session_state.language)
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(translate_text("← Back to Basic Info", st.session_state.language)):
            st.session_state.stage = "collect_basic_info"
            st.rerun()
    
    with col2:
        if st.button(translate_text("Generate Questions →", st.session_state.language), use_container_width=True):
            if check_exit_intent(tech_stack):
                st.success(translate_text("Thank you for your time. Goodbye!", st.session_state.language))
                st.session_state.stage = "ended"
                st.rerun()
            elif not tech_stack.strip():
                st.error(translate_text("Please describe your technical skills to continue.", st.session_state.language))
            elif len(tech_stack.strip()) < 10:
                st.warning(translate_text("Please provide more detailed information about your technical skills.", st.session_state.language))
            else:
                st.session_state.candidate_info["tech_stack"] = sanitize_input(tech_stack)
                
                # Generate questions
                with st.spinner(translate_text("Analyzing your skills and generating personalized questions...", st.session_state.language)):
                    questions = generate_technical_questions(
                        tech_stack=st.session_state.candidate_info["tech_stack"],
                        experience_level=st.session_state.candidate_info["experience"],
                        position=st.session_state.candidate_info["position"],
                        language=st.session_state.language
                    )
                
                st.session_state.current_question_set = questions
                st.session_state.questions_generated = True
                add_to_conversation_history("assistant", "Technical questions generated", {"tech_stack": tech_stack})
                st.session_state.stage = "show_questions"
                st.rerun()

# Stage 4: Show Questions and Collect Response
elif st.session_state.stage == "show_questions":
    st.subheader(translate_text("🎯 Your Personalized Technical Questions", st.session_state.language))
    
    # Candidate summary in a nice format
    st.markdown('<div class="candidate-info">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(translate_text("Candidate", st.session_state.language), st.session_state.candidate_info['name'])
    with col2:
        st.metric(translate_text("Position", st.session_state.language), st.session_state.candidate_info['position'])
    with col3:
        st.metric(translate_text("Experience", st.session_state.language), f"{st.session_state.candidate_info['experience']} years")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tech stack
    with st.expander(translate_text("🔧 Your Tech Stack", st.session_state.language)):
        st.write(st.session_state.candidate_info['tech_stack'])
    
    # Questions
    st.markdown(f"**{translate_text('Technical Interview Questions:', st.session_state.language)}**")
    st.markdown(st.session_state.current_question_set)
    
    st.markdown("---")
    
    # Optional response section
    st.markdown(f"**{translate_text('Your Response (Optional)', st.session_state.language)}**")
    st.markdown(translate_text(
        "You can provide brief answers or notes about your experience with these topics. This is optional but helps us understand your expertise better.",
        st.session_state.language
    ))
    
    candidate_response = st.text_area(
        translate_text("Your answers or additional comments:", st.session_state.language),
        height=150,
        help=translate_text("This information helps us better understand your expertise level", st.session_state.language)
    )
    
    # Sentiment analysis if response provided
    if candidate_response and not check_exit_intent(candidate_response):
        sentiment_analysis = analyze_sentiment(candidate_response, st.session_state.language)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(translate_text("Sentiment", st.session_state.language), sentiment_analysis["sentiment"])
        with col2:
            st.metric(translate_text("Confidence", st.session_state.language), sentiment_analysis["confidence"])
        with col3:
            st.info(f"📝 {sentiment_analysis['note']}")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button(translate_text("← Modify Tech Stack", st.session_state.language)):
            st.session_state.stage = "collect_tech_stack"
            st.rerun()
    
    with col2:
        if st.button(translate_text("🔄 Regenerate Questions", st.session_state.language)):
            with st.spinner(translate_text("Generating new questions...", st.session_state.language)):
                new_questions = generate_technical_questions(
                    tech_stack=st.session_state.candidate_info["tech_stack"],
                    experience_level=st.session_state.candidate_info["experience"],
                    position=st.session_state.candidate_info["position"],
                    language=st.session_state.language
                )
            st.session_state.current_question_set = new_questions
            st.rerun()
    
    with col3:
        if st.button(translate_text("✅ Complete Application", st.session_state.language), use_container_width=True):
            # Save the session data (in a real app, this would go to a database)
            session_data = {
                "candidate_info": st.session_state.candidate_info,
                "questions": st.session_state.current_question_set,
                "response": candidate_response,
                "sentiment": analyze_sentiment(candidate_response, st.session_state.language) if candidate_response else None,
                "timestamp": datetime.now().isoformat(),
                "language": st.session_state.language
            }
            
            add_to_conversation_history("user", candidate_response or "No response provided")
            add_to_conversation_history("assistant", "Application completed successfully", session_data)
            
            st.session_state.stage = "completion"
            st.rerun()

# Stage 5: Completion
elif st.session_state.stage == "completion":
    st.success(translate_text("🎉 Application Successfully Submitted!", st.session_state.language))
    
    completion_message = translate_text("""
    Thank you for completing your application with TalentScout!
    
    What happens next:
    
    1. Review Process: Our technical team will review your information and responses
    2. Initial Screening: We'll evaluate your technical background against our current openings
    3. Follow-up: If there's a match, our recruitment team will contact you within 3-5 business days
    4. Next Steps: You may be invited for a detailed technical interview or skills assessment
    
    Important Notes:
    - Keep an eye on your email (including spam folder)
    - Ensure your contact information is current
    - Feel free to update us if your circumstances change
    
    We appreciate your interest in opportunities with our partner companies!
    """, st.session_state.language)
    
    st.info(completion_message)
    
    # Data privacy notice
    privacy_notice = translate_text("""
    🔒 Privacy Notice: Your information is handled in compliance with GDPR and data privacy standards. 
    We only use your data for recruitment purposes and will not share it without your consent.
    """, st.session_state.language)
    
    st.caption(privacy_notice)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(translate_text("📥 Download Summary", st.session_state.language)):
            # Create a summary for download (simulated)
            summary = f"""
TalentScout Application Summary
==============================
Name: {st.session_state.candidate_info['name']}
Position: {st.session_state.candidate_info['position']}
Experience: {st.session_state.candidate_info['experience']} years
Tech Stack: {st.session_state.candidate_info['tech_stack']}
Submission Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            st.download_button(
                label=translate_text("Download as Text File", st.session_state.language),
                data=summary,
                file_name=f"talentscout_application_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button(translate_text("🆕 New Application", st.session_state.language)):
            # Reset for new application
            for key in ["candidate_info", "current_question_set", "questions_generated", "conversation_history"]:
                if key in st.session_state:
                    st.session_state[key] = {} if "info" in key else ([] if "history" in key else ("" if "question" in key else False))
            st.session_state.stage = "greeting"
            st.session_state.greeted = False
            st.rerun()

# Stage 6: Ended
elif st.session_state.stage == "ended":
    st.info(translate_text("Session ended. Thank you for using TalentScout! 🙏", st.session_state.language))
    
    if st.button(translate_text("Start New Session", st.session_state.language)):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"<center><small>{translate_text('TalentScout AI Hiring Assistant v2.0 | Powered by OpenAI GPT-4', st.session_state.language)}</small></center>",
    unsafe_allow_html=True
)
