from flask import *
from Scripts.Practise import tap_and_Select_v10

import os
UPLOAD_FOLDER = "\\venv\\LclSrc\\Scripts\\restApi\\static\\"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def upload():
    return render_template("upload.html")

@app.route('/success', methods=['GET','POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        return render_template("success.html", name=f.filename)


@app.route('/button')
def button():
    text,amount= tap_and_Select_v10.ocr_func('C:\\Users\\divya.malhotra\\PycharmProjects\\PracticeProject\\venv\\LclSrc\\Scripts\\restApi\\static\\easyday_bill.jpg',387,131,906,1004)
    print("text in main ", text)
    print("amount in main ", amount)
    return render_template('results.html',  value1=text,value2=amount)


if __name__ == '__main__':
    app.run(debug=True)