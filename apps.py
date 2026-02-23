from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("student_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        attendance = float(request.form["attendance"])

        new_data = pd.DataFrame({
            "Hours_Studied": [hours],
            "Attendance": [attendance]
        })

        prediction = model.predict(new_data)
        probability = model.predict_proba(new_data)

        pass_prob = round(probability[0][1] * 100, 2)
        fail_prob = round(probability[0][0] * 100, 2)

        if prediction[0] == 1:
            prediction_text = f"""
            PASS ✅ <br>
            Pass Probability: {pass_prob}% <br>
            Fail Probability: {fail_prob}%
            """
        else:
            prediction_text = f"""
            FAIL ❌ <br>
            Pass Probability: {pass_prob}% <br>
            Fail Probability: {fail_prob}%
            """

    # 🔥 THIS LINE MUST HAVE 4 SPACES BEFORE IT
    return render_template("index1.html", prediction=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)