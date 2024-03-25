const taskForm = (formName, doPoll, report) => {
    document.forms[formName].addEventListener("submit", (event) => {
      event.preventDefault()

      const fileInput = document.querySelector('input[type="file"]');
      
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
    const el = document.getElementById("process-result")

    if (data === null) {
      el.innerText = "submitted"
    } else if (!data["ready"]) {
      el.innerText = `${data["value"]["current"]} / ${data["value"]["total"]}`
    } else if (!data["successful"]) {
      el.innerText = "error, check console"
    } else {
      el.innerText = "✅ done"
    }
    console.log(data)
  })
