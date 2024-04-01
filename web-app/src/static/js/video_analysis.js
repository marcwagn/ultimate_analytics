import { drawPitchOutline, drawCircle, standardCoordsToCanvasCoords } from './render_pitch.js';

const taskForm = (formName, doPoll, report) => {
    document.forms[formName].addEventListener("submit", (event) => {
      event.preventDefault()

      const fileInput = document.querySelector('input[type="file"]');
      const videoPlayer = document.getElementById('videoplayer');
      
      const formData = new FormData(event.target)
      formData.append('file', fileInput.files[0]);

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
  const progressbar = document.getElementById("progressbar");
  const tacticalboard = document.getElementById("tacticalboard");
  const fileInput = document.querySelector('input[type="file"]');
  const videoPlayer = document.getElementById('videoplayer');


  if (data === null) {
    console.log("uploading...")
    progressbar.value = 0;
  } else if (!data["ready"]) {
    progressbar.value = data["value"]["status"]
  } else if (!data["successful"]) {
    console.log("error, check console")
  } else {
    progressbar.value = 1;

    var url = URL.createObjectURL(fileInput.files[0]);
    videoPlayer.src = url;
    videoPlayer.style.display = 'block';

    // get the coordinates of the players
    let person_coords = data["value"]["coordinates"]
    console.log(person_coords)

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

      // handle edge cases
      let personCoordsLength = Object.keys(person_coords).length;
      if (shown_frame >= personCoordsLength) {
        shown_frame = personCoordsLength - 1;
      } else if (shown_frame == 0) { 
        shown_frame = 1;
      }

      // render pitch on backbuffer
      drawPitchOutline(backBuffer);

      console.log(person_coords)
      console.log(person_coords.length)
      console.log(shown_frame)
    
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