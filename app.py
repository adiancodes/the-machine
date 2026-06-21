from flask import Flask, render_template, request, jsonify
from logic import extract_transcript

app = Flask(__name__)

@app.route('/')
def index():
    # Renders the HTML template from the "templates" folder
    return render_template('index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
        
    try:
        transcript = extract_transcript(url)
        return jsonify({'text': transcript})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the server on http://127.0.0.1:5000/
    app.run(debug=True)
