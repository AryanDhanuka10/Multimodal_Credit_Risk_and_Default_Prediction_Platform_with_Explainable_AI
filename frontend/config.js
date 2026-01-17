// frontend/config.js
const CONFIG = {
  BACKEND_URL: window.location.hostname === 'localhost' 
    ? 'http://localhost:8001'
    : 'https://credit-risk-api.hf.space'  // Will update this later
};
