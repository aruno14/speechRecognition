const express = require('express')
const port = 3000
const app = express()
app.use(express.static('html/'))
app.listen(port, () => console.log(`Aapp listening at http://localhost:${port}`));
