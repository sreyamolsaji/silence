<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />



# [Silence Compressor] 🎯


## Basic Details
### Team Name: [Kappens]


### Team Members
- Team Lead: [Catherine Jimmy Kappen] - [Muthoot Institute of Technology and Science]
- Member 2: [Sreyamol Saji] - [Muthoot Institute of Technology and Science]


### Project Description
[Silence Compressor is a real-time AI-assisted application designed to identify potentially awkward moments during conversations.

Instead of treating every period of silence as awkward, the system analyzes multiple conversational signals such as silence duration, audio activity, and facial/behavioral cues to estimate whether a particular pause may feel awkward.

The project is intentionally built around a problem that nobody really asked to solve: Can software detect when a conversation is becoming awkward?]

### The Problem (that doesn't exist)
[Awkward silence is difficult to detect using silence duration alone.

A short pause can be completely natural, while a longer pause does not necessarily mean that a conversation is awkward. The context of the conversation and the behavior of the people involved can make a significant difference.

Silence Compressor explores the ridiculous idea that a computer could try to understand these moments by combining multiple real-time signals instead of simply using a fixed silence threshold.

The system therefore attempts to answer:

"Is this actually an awkward silence, or are we just overthinking it?"]

### The Solution (that nobody asked for)
[The system monitors a conversation through the microphone and camera and processes the available signals in real time.

The application:

Captures audio from the microphone.
Detects periods of silence and conversational activity.
Captures visual information through the camera.
Uses facial/behavioral information as an additional contextual signal.
Combines the available signals to estimate the likelihood of an awkward moment.
Displays the result through an interactive Streamlit interface.
Provides visual feedback and a timeline of detected conversational events.

The goal is not to scientifically prove that a conversation is awkward, but to create a technically functional and unnecessarily complicated way of measuring something humans normally notice instinctively.]

## Technical Details
### Technologies/Components Used
For Software:
- [Python 3.14]
-[Streamlit – Web-based interactive interface]
-[OpenCV – Camera/video processing]
NumPy – Numerical and signal processing operations
PyAudio / audio input libraries – Microphone and audio capture
Python audio-processing libraries – Audio analysis and silence detection
Visual Studio Code – Development environment
PowerShell – Environment setup and execution
Git & GitHub – Version control and project repository



### Implementation
For Software:
# Installation
[Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>

Create a virtual environment:

python -m venv venv

Activate the virtual environment in PowerShell:

.\venv\Scripts\Activate.ps1

Install the required dependencies:

pip install -r requirements.txt]

# Run
[Start the Streamlit application:
streamlit run app.py
The application will open in the browser.
Allow access to the microphone and camera when requested.]

### Project Documentation
For Software:

# Screenshots (Add at least 3)
![<img width="1597" height="907" alt="img1" src="https://github.com/user-attachments/assets/f48ce2f6-8a95-4820-af75-2ea6a1a4507a" />
](Add screenshot 1 here with proper name)
*Add caption explaining what this shows*

![<img width="1069" height="624" alt="img 2" src="https://github.com/user-attachments/assets/0712c6ba-6157-456e-be1e-cd0ff6d3b343" />
](Add screenshot 2 here with proper name)
*Add caption explaining what this shows*

![<img width="1244" height="801" alt="img 3" src="https://github.com/user-attachments/assets/d1edf268-2cc6-49e7-8eb2-2ea64fa64f1d" />
](Add screenshot 3 here with proper name)
*Add caption explaining what this shows*

# Diagrams
![Workflow of Silence Compressor: microphone and camera input → signal processing → feature extraction → multi-signal analysis → awkwardness estimation → Streamlit visualization.

System Workflow
             ┌──────────────────┐
             │    Conversation  │
             └────────┬─────────┘
                      │
             ┌────────┴─────────┐
             │                  │
             ▼                  ▼
      ┌─────────────┐    ┌─────────────┐
      │ Microphone  │    │   Camera    │
      └──────┬──────┘    └──────┬──────┘
             │                  │
             ▼                  ▼
      ┌─────────────┐    ┌─────────────┐
      │ Audio       │    │ Visual      │
      │ Processing  │    │ Processing  │
      └──────┬──────┘    └──────┬──────┘
             │                  │
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ Feature Analysis │
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ Awkwardness      │
             │ Estimation       │
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ Streamlit        │
             │ Dashboard        │
             └──────────────────┘](Add your workflow/architecture diagram here)
*Workflow of Silence Compressor — real-time audio and visual inputs are processed to detect silence and behavioral cues, which are combined to estimate conversational awkwardness and display the results through the Streamlit dashboard.*


## Team Contributions
- [Catherine Jimmy Kappen]: [ Project ideation, application design, interface development, testing, documentation, and project presentation.]
- [Sreyamol Saji]: [ Audio-processing implementation, silence detection, real-time signal analysis, application development, debugging, testing, GitHub integration, and technical documentation.]


---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)



