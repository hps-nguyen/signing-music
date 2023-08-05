# Sign Language Recognition for Computer Music Performance
## UCSD SIPP 2023 Project

### Abstract
"Sign Language Recognition for Computer Music Performance" is a personal exploration that seeks to combine Computer Vision, Machine Learning, and Data Processing in the Python environment. Driven by a curiosity to harness technology in novel ways, this project plays with the idea of converting gestures, some inspired by American Sign Language signs, into directives for computer music performance. By leveraging tools like OpenCV and MediaPipe libraries, it attempts to translate hand gestures into digital instructions, aiming to bridge the gap between human expressiveness and the vast potential of artificial intelligence in artistic pursuits. This endeavor represents not just a technical experiment, but also a journey of passion, discovery, and an eagerness to learn.

---

### Introduction
Hello! My journey has always been driven by a deep appreciation for both music and data science. While they might seem contrasting, these domains uniquely intersect in many ways. Music captures the essence of human emotion, while data science provides tools to analyze and shape experiences. The project you're about to explore exemplifies this intriguing synergy.

### Why This Project?
Though not stemming from an engineering background, I'm intrigued by its essence. The world of music is not devoid of engineering; think of the intricacies behind each musical note. With digital advancements, the essence of music is transforming, making it more about the ideas and less about the instrument.

### Technical Overview
- Foundation: Python, chosen for its diverse libraries.
- Video Capture: Done through the OpenCV library, capturing real-time gestures.
- Hand Detection: Enabled by the MediaPipe library, vital for 2D hand coordinate extraction.
- Gesture Classification: Entrusted to the scikit-learn library. Each frame gets labeled, guiding the project's actions.
- Despite the assistance from libraries, challenges were aplenty. The differentiation of gestures for both hands, for instance. The right hand, in my project, indicates pitch modulation, whereas the left signs the pitch.

### Delving Deeper
Gesture Classification and Processing: Two classifiers, each specifically trained for the left and right hand, are central to this system. We harness the power of the random forest algorithm for the primary training, while the isolation forest algorithm acts as a gatekeeper, identifying and sidelining unrecognizable gestures. The gestures not only convey musical notes but also nuances like volume and vibration, dictated by the relative positioning of the hands.

Musical Interpretation with FluidSynth: The detected gestures serve as the foundation upon which the music-playing module is built. Each frame's parameters are interpreted and translated into music, with the aid of a specialized music library and MIDI messages. Additionally, the incorporation of the fluidsynth software synthesizer enriches the musical output. It not only processes the MIDI signals but offers the flexibility of real-time track and instrument adjustments, ensuring the resultant music is as dynamic and expressive as the gestures guiding it.

### Acknowledgements
A special note of gratitude goes to the YouTube channel "Computer vision engineer" for their invaluable tutorial on sign language detection. Their video tutorial and accompanying code have been instrumental in my understanding of computer vision and the practical application of machine learning. Their content not only educated but also inspired facets of this project.

### Concluding Remarks
This is an ongoing endeavor. There's potential for refinement in areas like latency reduction, pitch range expansion, and overall reliability enhancement. However, the current version stands as a testament to the fusion of human creativity with technological innovation.

Feel free to share your feedback or raise any queries!
