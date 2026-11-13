
from flask import Flask, render_template, request
import joblib
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime

from io import BytesIO

from flask import send_file

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table
)
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "house_price_model.pkl")

LOCATIONS = [
    "Chennai",
    "Coimbatore",
    "Madurai",
    "Tirunelveli",
    "Tenkasi"
]

DATABASE = BASE_DIR / "predictions.db"


# Create database
def init_db():
    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT,
            area INTEGER,
            bedrooms INTEGER,
            bathrooms INTEGER,
            age INTEGER,
            predicted_price REAL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


# Get prediction history
def get_history():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT 10
    """).fetchall()

    conn.close()
    return rows


# Calculate city average prices
def get_location_prices():
    data = pd.read_csv(BASE_DIR / "house_data.csv")

    averages = (
        data.groupby("location")["price_inr"]
        .mean()
        .reindex(LOCATIONS)
        .fillna(0)
    )

    return {
        city: round(float(price))
        for city, price in averages.items()
    }


# Save prediction
def save_prediction(location, area, bedrooms,
                    bathrooms, age, price):

    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        INSERT INTO predictions
        (location, area, bedrooms, bathrooms,
         age, predicted_price, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        location,
        area,
        bedrooms,
        bathrooms,
        age,
        price,
        datetime.now().strftime("%d-%m-%Y %H:%M")
    ))

    conn.commit()
    conn.close()


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    error = None

    if request.method == "POST":

        try:
            location = request.form["location"]
            area = int(request.form["area_sqft"])
            bedrooms = int(request.form["bedrooms"])
            bathrooms = int(request.form["bathrooms"])
            age = int(request.form["property_age"])

            if location not in LOCATIONS:
                raise ValueError("Select a valid location.")

            if not 300 <= area <= 10000:
                raise ValueError(
                    "Area must be between 300 and 10000."
                )

            if not 1 <= bedrooms <= 10:
                raise ValueError("Invalid bedroom count.")

            if not 1 <= bathrooms <= 10:
                raise ValueError("Invalid bathroom count.")

            if not 0 <= age <= 100:
                raise ValueError("Invalid property age.")

            house = pd.DataFrame([{
                "location": location,
                "area_sqft": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "property_age": age
            }])

            prediction = round(
                float(model.predict(house)[0])
            )

            save_prediction(
                location, area, bedrooms,
                bathrooms, age, prediction
            )

        except (ValueError, KeyError) as e:
            error = str(e) or "Please enter valid details."

    history = get_history()
    location_prices = get_location_prices()

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        locations=LOCATIONS,
        history=history,
        location_prices=location_prices
    )


@app.route("/download-report", methods=["POST"])
def download_report():

    location = request.form.get("location", "")
    area = request.form.get("area_sqft", "")
    bedrooms = request.form.get("bedrooms", "")
    bathrooms = request.form.get("bathrooms", "")
    age = request.form.get("property_age", "")
    prediction = request.form.get("prediction", "")

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()
    content = []

    content.append(
        Paragraph("HomeValue AI", styles["Title"])
    )

    content.append(
        Paragraph("Property Price Estimation Report",
                  styles["Heading2"])
    )

    content.append(Spacer(1, 25))

    data = [
        ["Property Details", "Value"],
        ["Location", location],
        ["Area", str(area) + " sq.ft"],
        ["Bedrooms", str(bedrooms)],
        ["Bathrooms", str(bathrooms)],
        ["Property Age", str(age) + " years"],
        ["Estimated Price", "Rs. " + str(prediction)]
    ]

    table = Table(data, colWidths=[200, 200])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.blue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 12),
    ]))

    content.append(table)
    content.append(Spacer(1, 30))

    content.append(
        Paragraph(
            "Generated by HomeValue AI - Smart Real Estate Tool",
            styles["Normal"]
        )
    )

    doc.build(content)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="HomeValue_Property_Report.pdf",
        mimetype="application/pdf"
    )

if __name__ == "__main__":

    init_db()

    app.run(debug=True)
