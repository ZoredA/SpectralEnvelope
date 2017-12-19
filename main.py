from appJar import gui


app = gui("Frequency Estimation and Envelope Estimation")
app.setSticky("news")
app.setExpand("both")
app.setFont(20)

app.addButtons(["Open File",  "Generate Wave"], "row=0" )

app.addLabelEntry("Open File")


app.addLabel("title", "Welcome to appJar")
app.setLabelBg("title", "red")

app.go()
