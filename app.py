import streamlit as st
import tempfile
import cv2
import os
from dotenv import load_dotenv
import google.generativeai as genai
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from fpdf import FPDF  # Import FPDF library for PDF creation
import re

# Load environment variables
load_dotenv()

# Configure the Google Generative AI API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the prompt for the AI model
prompt = """You are a video summarizer. You will be taking the transcript text
and summarizing the entire video, providing the important summary in points
within 1500 words. Please provide the summary of the text given here: """

# Function to extract transcript details from an uploaded video using speech recognition
def extract_transcript_details(video_file):
    try:
        st.write("Extracting transcript from video...")

        # Use moviepy to extract audio from video and save as WAV
        clip = VideoFileClip(video_file)
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        clip.audio.write_audiofile(temp_audio.name)
        clip.close()

        # Perform speech recognition on the extracted audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio.name) as source:
            audio = recognizer.record(source)

        transcript_text = recognizer.recognize_google(audio)

        st.write("Transcript extracted successfully")
        return transcript_text

    except Exception as e:
        st.error(f"An error occurred while fetching the transcript: {e}")
        return None

# Function to generate content based on the transcript using Google Gemini Pro
def generate_gemini_content(transcript_text, prompt):
    try:
        st.write("Generating summary with Google Gemini Pro")
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript_text)
        st.write("Summary generated successfully")
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating the summary: {e}")
        return None

# Function to extract frames at specific intervals from uploaded video
def extract_frames_video(video_file, interval=30):
    try:
        st.write("Extracting frames from uploaded video...")

        clip = VideoFileClip(video_file)
        fps = clip.fps
        duration = clip.duration
        frame_interval = int(fps * interval)

        frames = []
        for i in range(0, int(duration), interval):
            frame = clip.get_frame(i)
            frames.append((frame, i))

        st.write("Frames extracted successfully")
        return frames

    except Exception as e:
        st.error(f"An error occurred while extracting frames: {e}")
        return None

# Function to generate PDF from detailed notes
def generate_combined_pdf(summary_sections, frames, save_folder):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Add title
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Detailed Notes from Video Summary", ln=True, align='C')

        # Add sections and corresponding images
        pdf.set_font("Arial", size=12)
        for section_idx, section in enumerate(summary_sections):
            pdf.add_page()
            pdf.cell(200, 10, txt=f"Summary Section {section_idx + 1}", ln=True, align='C')
            pdf.multi_cell(200, 10, txt=section)

            # Add image if available
            if section_idx < len(frames):
                frame, timestamp = frames[section_idx]
                temp_file = f"{save_folder}/frame_{section_idx}.jpg"
                cv2.imwrite(temp_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                pdf.image(temp_file, x=10, y=pdf.get_y() + 10, w=180)

        pdf_file = f"{save_folder}/detailed_notes.pdf"
        pdf.output(pdf_file)
        return pdf_file

    except Exception as e:
        st.error(f"An error occurred while generating PDF: {e}")
        return None

# Streamlit application title and file upload
st.title("Video Transcript to Detailed Notes Converter")
uploaded_file = st.file_uploader("Upload a video (MP4)", type=["mp4"])

# Button to process uploaded video and generate detailed notes
if st.button("Process Video"):
    if uploaded_file is not None:
        st.write("Processing uploaded video...")

        # Save the uploaded video to a temporary file
        try:
            temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_video.write(uploaded_file.read())
            temp_video.close()

            # Extract transcript and generate summary
            transcript_text = extract_transcript_details(temp_video.name)
            if transcript_text:
                summary = generate_gemini_content(transcript_text, prompt)
                if summary:
                    st.markdown("## Detailed Notes:")
                    st.write(summary)

                    # Extract frames and display relevant images
                    frames = extract_frames_video(temp_video.name)
                    if frames:
                        st.markdown("## Details in Explanation:")
                        
                        # Split summary based on the pattern
                        summary_sections = re.split(r"(\*\*.+?\n\n)", summary)
                        summary_sections = ["".join(summary_sections[i:i+2]) for i in range(0, len(summary_sections), 2) if i+1 < len(summary_sections)]
                        st.write("summary section", summary_sections)
                        # Specify folder to save PDF and images
                        save_folder = "./output"
                        os.makedirs(save_folder, exist_ok=True)

                        # Generate combined PDF with all sections and images
                        pdf_file = generate_combined_pdf(summary_sections, frames, save_folder)
                        if pdf_file:
                            st.markdown(f"### [Download Detailed Notes as PDF]({pdf_file})")

                            # Provide a download button for the combined PDF
                            with open(pdf_file, "rb") as file:
                                st.download_button(
                                    label="Download Combined Detailed Notes PDF",
                                    data=file,
                                    file_name="detailed_notes.pdf",
                                )

                            # Display section summaries in the UI
                            st.markdown("## Section Summaries:")
                            for idx, section in enumerate(summary_sections):
                                st.markdown(f"### Section {idx + 1}:")
                                st.write(section)

                            # Display frames in the UI
                            st.markdown("## Extracted Frames:")
                            for frame_idx, (frame, timestamp) in enumerate(frames):
                                temp_file = f"{save_folder}/frame_{frame_idx}.jpg"
                                cv2.imwrite(temp_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                                st.image(temp_file, caption=f"Frame at {timestamp} seconds")

        except Exception as e:
            st.error(f"An error occurred: {e}")

        finally:
            # Clean up temporary files (if any)
            if os.path.exists(temp_video.name):
                os.remove(temp_video.name)
