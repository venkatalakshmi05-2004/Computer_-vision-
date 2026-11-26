# Computer Vision Gesture & Face Detection — Mobile Web Edition

A mobile-friendly browser version of the original Python OpenCV project. It uses the phone camera through the browser and MediaPipe Tasks Vision for face landmarks and hand landmarks.

## Features
- Live face detection with bounding boxes
- Hand landmark tracking and approximate finger counting (0–5)
- Front/back camera switch
- Mobile-friendly interface
- Runs in a modern browser; no Python installation required

## Run on an Android phone

Camera access in Chrome requires a secure context (HTTPS) or localhost. The easiest way to host this project is GitHub Pages:

1. Upload `index.html`, `style.css`, `app.js`, and this `README.md` to the root of your GitHub repository.
2. Open the repository on GitHub.
3. Go to **Settings → Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Select branch **main** and folder **/(root)**, then save.
6. Wait for GitHub Pages to publish the site. Open the Pages URL shown in Settings → Pages on your Android phone using Chrome.
7. Tap **Start Camera** and allow camera permission.

## Local testing
Serve the folder through a local HTTP server (not by opening `index.html` as a file). For example:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000` in a browser on the same computer.

## Notes
- The first start downloads the MediaPipe JavaScript runtime and model files, so an internet connection is required.
- Browser camera access requires HTTPS (GitHub Pages provides HTTPS).
- Face and hand analysis runs in the browser. Camera frames are not uploaded by this app.
- Finger counting is an estimate based on hand landmarks and may be inaccurate for rotated hands, crossed fingers, or some camera angles.
- This is a JavaScript/MediaPipe browser adaptation; it does not execute the original Python `main.py` file.
