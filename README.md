# 🤖 TalentScout - AI Hiring Assistant

An intelligent hiring assistant chatbot designed for TalentScout, a fictional recruitment agency specializing in technology placements. This application assists in the initial screening of candidates by gathering essential information and generating relevant technical questions based on their declared tech stack.

## 📋 Project Overview

TalentScout is a sophisticated AI-powered hiring assistant that streamlines the candidate screening process through:

- Intelligent Information Gathering: Collects essential candidate details including personal information, experience, and technical skills
- Dynamic Question Generation: Creates personalized technical interview questions based on the candidate's tech stack and experience level
- Enhanced User Experience: Features multilingual support, sentiment analysis, and robust conversation handling
- Data Privacy Compliance: Implements security measures and GDPR compliance for sensitive candidate information

### Key Features

#### Core Functionality
- ✅ Professional Greeting & Context Setting: Clear introduction with purpose explanation
- ✅ Comprehensive Data Collection: Gathers all required candidate information
- ✅ Tech Stack-Based Question Generation: Creates 4 tailored technical questions
- ✅ Graceful Exit Handling: Supports multiple exit keywords and conversation ending
- ✅ Robust Fallback Mechanism: Handles unexpected inputs while staying on purpose

#### Bonus Features (Advanced Implementation)
- 🌟 Multilingual Support: Interface available in 6 languages (English, Spanish, French, German, Hindi, Chinese)
- 🌟 Sentiment Analysis: Analyzes candidate responses with confidence scoring
- 🌟 Enhanced UI/UX: Custom styling, responsive design, and interactive elements
- 🌟 Data Security: Input sanitization, validation, and privacy compliance
- 🌟 Context Management: Maintains conversation history and session state

## 🚀 Installation Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- OpenAI API key

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd project
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration

Create a `.env` file in the project root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

Note: Replace `your_openai_api_key_here` with your actual OpenAI API key. You can obtain one from [OpenAI's website](https://platform.openai.com/).

### Step 5: Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## 📖 Usage Guide

### Getting Started

1. Launch the Application: Run the Streamlit app using the command above
2. Select Language: Choose your preferred language from the sidebar (optional)
3. Begin Interview: Click "Yes" to start the hiring process

### Application Flow

#### Stage 1: Greeting
- Welcome message with clear purpose explanation
- Option to begin or exit the conversation

#### Stage 2: Basic Information Collection
- Required Fields: Full name, email, phone number, desired position, years of experience
- Optional Fields: Current location
- Validation: Email format and phone number validation
- Navigation: Continue to tech stack or exit

#### Stage 3: Technical Stack Declaration
- Input: Detailed description of technical skills and technologies
- Examples: Provided for different roles (Frontend, Backend, Full Stack, Data Science)
- Validation: Minimum detail requirements
- Navigation: Back to basic info, continue to questions, or exit

#### Stage 4: Question Generation & Response
- Display: Personalized technical questions based on experience and tech stack
- Optional Response: Candidate can provide answers or additional comments
- Sentiment Analysis: Real-time analysis of candidate responses
- Actions: Modify tech stack, regenerate questions, or complete application

#### Stage 5: Completion
- Summary: Application submission confirmation
- Next Steps: Clear explanation of the hiring process
- Privacy Notice: GDPR compliance information
- Options: Download summary or start new application

### Advanced Features

- Multilingual Support: Switch languages at any time using the sidebar
- Session Management: Unique session IDs and conversation history tracking
- Fallback Handling: Smart responses for off-topic or unclear inputs
- Data Security: All inputs are sanitized and validated

## 🔧 Technical Details

### Architecture

The application follows a modular, state-driven architecture:

```
streamlit_app.py
├── Configuration & Constants
├── Security & Data Privacy Functions
├── Session State Management
├── Translation & Multilingual Support
├── Enhanced Prompt Engineering
├── Fallback Mechanism
├── Sentiment Analysis
├── Conversation Management
└── Main Application Flow (6 stages)
```

### Libraries & Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.47.1 | Web application framework |
| openai | 1.98.0 | Large Language Model integration |
| python-dotenv | 1.1.1 | Environment variable management |
| python | 3.8+ | Core programming language |

### Model Details

- Primary Model: OpenAI GPT-4
- Temperature Settings: 
  - Question Generation: 0.7 (creative)
  - Translation: 0.2 (precise)
  - Sentiment Analysis: 0.1 (consistent)
- Token Limits: Optimized for cost and performance
- Fallback Handling: Graceful degradation on API failures

### Data Flow

1. Input Sanitization: All user inputs are cleaned and validated
2. State Management: Session state maintains context across interactions
3. API Communication: Structured prompts sent to OpenAI GPT-4
4. Response Processing: AI responses are parsed and formatted
5. Security Compliance: Sensitive data is hashed for privacy

### Security Features

- Input Sanitization: Prevents injection attacks
- Data Validation: Email and phone number format checking
- Privacy Compliance: GDPR-compliant data handling
- Session Security: Unique session IDs and secure state management

## 🎯 Prompt Design

### Core Purpose Definition

The application implements a strict purpose definition to ensure the AI stays focused on hiring tasks:

```python
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
```

### Question Generation Strategy

The technical question generation uses sophisticated prompt engineering:

#### Experience-Based Difficulty Scaling
```python
if experience_level <= 2:
    difficulty = "beginner to intermediate"
    depth = "fundamental concepts and basic implementation"
elif experience_level <= 5:
    difficulty = "intermediate to advanced" 
    depth = "design patterns, best practices, and problem-solving"
else:
    difficulty = "advanced to expert"
    depth = "architecture, optimization, and leadership scenarios"
```

#### Structured Prompt Template
```python
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
```

### Fallback Mechanism Design

The fallback system uses context-aware prompting:

```python
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
```

### Multilingual Implementation

Translation prompts preserve technical terminology and professional tone:

```python
prompt = f"""Translate the following text to {target_lang}. 
Keep the tone professional and preserve any technical terms:

{text}"""
```

## 🛠 Challenges & Solutions

### Challenge 1: Maintaining Conversation Context

Problem: Ensuring the AI remembers previous interactions and maintains coherent conversation flow.

Solution: 
- Implemented comprehensive session state management
- Created conversation history tracking with metadata
- Used structured data flow to maintain context across stages

```python
def add_to_conversation_history(role: str, content: str, metadata: Optional[Dict] = None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content,
        "metadata": metadata or {}
    }
    st.session_state.conversation_history.append(entry)
```

### Challenge 2: Preventing AI from Deviating from Purpose

Problem: Ensuring the chatbot stays focused on hiring tasks and doesn't engage in general conversation.

Solution:
- Defined strict CORE_PURPOSE guidelines in all prompts
- Implemented robust fallback mechanism for off-topic inputs
- Added purpose-focused response templates

### Challenge 3: Handling Diverse Tech Stacks

Problem: Generating relevant questions for various technology combinations and experience levels.

Solution:
- Created experience-based difficulty scaling algorithm
- Implemented structured prompt templates with multiple requirements
- Added validation for minimum tech stack detail

### Challenge 4: Data Privacy and Security

Problem: Handling sensitive candidate information securely and complying with privacy standards.

Solution:
- Implemented input sanitization and validation functions
- Added data hashing for sensitive information
- Created GDPR-compliant data handling procedures
- Added security warnings and privacy notices

```python
def sanitize_input(text: str) -> str:
    if not text:
        return ""
    # Remove potentially harmful characters
    sanitized = re.sub(r'[<>"\';()&]', '', text)
    # Limit length to prevent abuse
    sanitized = sanitized[:500]
    return sanitized.strip()
```

### Challenge 5: Multilingual Support Implementation

Problem: Providing seamless multilingual experience without compromising functionality.

Solution:
- Implemented dynamic translation system using OpenAI API
- Created fallback to English when translation fails
- Preserved technical terminology across languages
- Added language persistence in session state

### Challenge 6: User Experience Optimization

Problem: Creating an intuitive and engaging interface that guides users smoothly through the process.

Solution:
- Designed progressive disclosure with clear stages
- Added helpful examples and guidance text
- Implemented responsive design with custom CSS
- Created interactive feedback with metrics and progress indicators

### Challenge 7: Error Handling and Robustness

Problem: Gracefully handling API failures, network issues, and unexpected user inputs.

Solution:
- Added comprehensive try-catch blocks around API calls
- Implemented fallback responses for service failures
- Created graceful degradation for non-critical features
- Added user-friendly error messages

## 🚀 Deployment Considerations

### Local Deployment
The application is configured for local deployment with minimal setup requirements.

### Cloud Deployment (Bonus)
For cloud deployment, consider:
- Streamlit Cloud: Direct GitHub integration
- AWS/GCP: Container deployment with proper environment variable management
- Security: Enhanced API key management and HTTPS configuration

## 🔄 Future Enhancements

1. Database Integration: Store candidate information persistently
2. Advanced Analytics: Detailed reporting on candidate interactions
3. Integration APIs: Connect with existing HR systems
4. Voice Interface: Add speech-to-text capabilities
5. Advanced AI Features: Implement GPT-4 vision for resume analysis

## 📝 License

This project is developed as part of an AI/ML internship assignment for TalentScout.

## 🤝 Support

### Common Issues
If you encounter the "Translation service temporarily unavailable" error or other API-related issues, please check the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide for detailed solutions.

### Quick Fixes
- API Issues: Use the "🔧 Test API Connection" button in the sidebar
- Translation Problems: The app includes fallback translation for core functionality
- Model Access: App automatically tries multiple models (gpt-4o → gpt-4 → gpt-3.5-turbo)

For technical support or questions about the implementation, please refer to the documentation or contact the development team.

---

TalentScout AI Hiring Assistant v2.0 | Powered by OpenAI GPT-4 | Built with Streamlit 