import { drawPitchOutline, drawCircle, standardCoordsToCanvasCoords } from './render_pitch.js';

const $ = (selector) => document.querySelector(selector);

const taskForm = (formName, doPoll, report) => {
    document.forms[formName].addEventListener("submit", (event) => {
      event.preventDefault()

      const fileInput = document.querySelector('input[type="file"]');
      const videoPlayer = document.getElementById('videoplayer');
      
      const formData = new FormData(event.target)
      formData.append('file', fileInput.files[0]);

      var url = URL.createObjectURL(fileInput.files[0]);
      videoPlayer.src = url;
      videoPlayer.style.display = 'block';

      fetch(event.target.action, {
        method: "POST",
        body: formData
      })
        .then(response => response.json())
        .then(data => {
          report(null)

          const poll = () => {
            fetch(`/tasks/result/${data["result_id"]}`)
              .then(response => response.json())
              .then(data => {
                report(data)

                if (!data["ready"]) {
                  setTimeout(poll, 500)
                } else if (!data["successful"]) {
                  console.error(formName, data)
                }
              })
          }

          if (doPoll) {
            poll()
          }
        })
    })
}

taskForm("video-upload-form", true, data => {
  const progressbar = document.getElementById("progressbar")
  const tacticalboard = document.getElementById("tacticalboard")

  if (data === null) {
    console.log("uploading...")
  } else if (!data["ready"]) {
    progressbar.value = data["value"]["status"]
  } else if (!data["successful"]) {
    console.log("error, check console")
  } else {
    progressbar.value = 1;

    // get the coordinates of the players
    let person_coords = data["value"]["coordinates"]

    // create a backbuffer canvas to draw the pitch on
    let backBuffer = document.createElement('canvas');
    backBuffer.width = tacticalboard.width;
    backBuffer.height = tacticalboard.height;
    let backBufferContext = backBuffer.getContext('2d');

    const updateCanvas = (now, metadata) => {
      let shown_frame = Math.floor(metadata["mediaTime"] * 30)

      // clear backbuffer
      backBufferContext.fillStyle = 'rgba(255, 255, 255, 0)';
      backBufferContext.fillRect(0, 0, backBuffer.width, backBuffer.height);

      // render pitch on backbuffer
      drawPitchOutline(backBuffer);
    
      // render players on backbuffer
      for (let person of person_coords[shown_frame]) {
        if (person.cls == 0) {

          const {x, y} = standardCoordsToCanvasCoords(person.x, person.y, backBuffer);

          if (person.team == 0) {
            drawCircle(backBufferContext, x, y + shown_frame, 5, 'black')
          } else {
            drawCircle(backBufferContext, x, y + shown_frame, 5, 'yellow')
          }
        }
      }
      // copy backbuffer to screen
      tacticalboard.getContext('2d').drawImage(backBuffer, 0, 0);
      // request next frame
      videoplayer.requestVideoFrameCallback(updateCanvas);
    };

  videoplayer.requestVideoFrameCallback(updateCanvas);
  }
})