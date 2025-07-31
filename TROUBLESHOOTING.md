# 🔧 TalentScout Troubleshooting Guide

## Common Issues and Solutions

### ❌ "Translation service temporarily unavailable" Error

This error occurs when the OpenAI API calls for translation fail. Here are the solutions in order of likelihood:

#### 1. **API Key Issues** (Most Common)
**Problem**: Invalid or missing OpenAI API key
**Solutions**:
- Check your `.env` file contains: `OPENAI_API_KEY=your_actual_api_key_here`
- Verify your API key is valid at [OpenAI Platform](https://platform.openai.com/api-keys)
- Ensure there are no extra spaces or quotes around the API key
- Make sure the .env file is in the same directory as streamlit_app.py

#### 2. **Model Access Issues**
**Problem**: Your API key doesn't have access to GPT-4
**Solutions**:
- The app now automatically tries multiple models: gpt-4o → gpt-4 → gpt-3.5-turbo
- Make sure you have access to at least gpt-3.5-turbo
- Check your OpenAI account tier and model permissions

#### 3. **API Credits/Billing**
**Problem**: No available API credits
**Solutions**:
- Check your OpenAI account billing at [OpenAI Billing](https://platform.openai.com/account/billing)
- Add payment method or credits to your account
- Verify your usage limits haven't been exceeded

#### 4. **Rate Limiting**
**Problem**: Too many API requests in a short time
**Solutions**:
- Wait a few minutes before trying again
- The app will show a specific rate limit warning
- Consider upgrading your OpenAI plan for higher limits

#### 5. **Network/Connection Issues**
**Problem**: Network connectivity or firewall blocking API calls
**Solutions**:
- Check your internet connection
- Try from a different network
- Verify corporate firewalls aren't blocking OpenAI API endpoints

### 🔍 How to Diagnose the Issue

#### Using the Built-in API Test
1. Open the app sidebar
2. Click "🔧 Test API Connection"
3. Check the results:
   - ✅ **Success**: Your API is working, translation should work
   - ❌ **API Error**: Follow the specific error message guidance

#### Manual Testing
Create a simple test file to verify your API key:

```python
import openai
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("✅ API working:", response.choices[0].message.content)
except Exception as e:
    print("❌ API error:", e)
```

### 🌐 Translation Fallback System

**Good News**: Even if AI translation fails, the app includes a basic translation system for common phrases in Spanish and French. The core functionality will continue to work in English.

### 📝 Environment Setup Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All requirements installed: `pip install -r requirements.txt`
- [ ] `.env` file created with valid `OPENAI_API_KEY`
- [ ] OpenAI account has available credits
- [ ] Network allows connections to api.openai.com

### 🆘 Still Having Issues?

1. **Check the console/terminal** where you're running `streamlit run streamlit_app.py` for detailed error messages
2. **Try running in English only** first to verify core functionality
3. **Test with a fresh virtual environment** to rule out dependency conflicts
4. **Verify your OpenAI account status** and billing information

### 📞 Error Messages and Their Meanings

| Error Message | Cause | Solution |
|---------------|-------|----------|
| "API key issue" | Invalid/missing API key | Check .env file and API key validity |
| "Translation rate limit reached" | Too many requests | Wait and try again later |
| "No available models" | Model access issues | Verify model permissions in OpenAI account |
| "AI translation unavailable" | General API failure | Check network and API status |

### 🔄 Quick Reset Steps

If all else fails, try these reset steps:
1. Close the Streamlit app
2. Delete any `.streamlit` folder in your project
3. Check your `.env` file one more time
4. Restart: `streamlit run streamlit_app.py`
5. Use the API test button in the sidebar

---

**Remember**: The app is designed to gracefully handle translation failures. Even if translation doesn't work, all core hiring assistant functionality will continue to operate in English. 