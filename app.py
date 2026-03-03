from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

app = Flask(__name__)

# Load dataset
df = pd.read_csv("data/jobs.csv")

# Create ML model
vectorizer = CountVectorizer()
vectorizer.fit(df["Skills"])


def recommend_jobs(user_skills, selected_industry, selected_experience):

    fallback_message = None

    # Step 1: Apply Industry Filter
    filtered_df = df.copy()

    if selected_industry != "All":
        filtered_df = filtered_df[
            filtered_df["Industry"] == selected_industry
        ]

    # Step 2: Apply Experience Filter
    if selected_experience != "All":
        exp_filtered = filtered_df[
            filtered_df["Experience_Level"] == selected_experience
        ]

        # If no jobs found for selected experience
        if exp_filtered.empty:
            fallback_message = (
                f"No {selected_experience} level jobs found"
                f" in {selected_industry}. Showing closest available matches."
            )
        else:
            filtered_df = exp_filtered

    # Step 3: If still empty → use full dataset
    if filtered_df.empty:
        fallback_message = "No jobs found for selected filters. Showing general recommendations."
        filtered_df = df.copy()

    # Step 4: ML Matching
    filtered_matrix = vectorizer.transform(filtered_df["Skills"])
    user_vector = vectorizer.transform([user_skills])
    similarity = cosine_similarity(user_vector, filtered_matrix)

    filtered_df["Score"] = similarity[0]

    # If all similarity scores are zero
    if filtered_df["Score"].sum() == 0:
        fallback_message = "No skill match found. Showing general recommendations."
        filtered_df["Score"] = 0.01  # small value to keep graph visible

    # Step 5: Sort results
    result = filtered_df.sort_values(
        by="Score",
        ascending=False
    ).head(5)

    result["Score"] = (result["Score"] * 100).round(2)

    # Step 6: Create Graph
    fig = px.bar(
        result,
        x="Job_Role",
        y="Score",
        title="Recommended Jobs Based on Skill Match (%)"
    )

    graph = fig.to_html(full_html=False)

    return result[
        ["Job_Role", "Experience_Level",
         "Salary_Range", "Industry", "Score"]
    ], graph, fallback_message


@app.route("/", methods=["GET", "POST"])
def home():

    table = None
    graph = None
    message = None

    industries = ["All"] + sorted(df["Industry"].unique())
    experiences = ["All"] + sorted(df["Experience_Level"].unique())

    if request.method == "POST":
        skills = request.form["skills"]
        industry = request.form["industry"]
        experience = request.form["experience"]

        table_df, graph, message = recommend_jobs(
            skills, industry, experience
        )

        if not table_df.empty:
            table = table_df.to_html(classes="table", index=False)
        else:
            table = "<p>No jobs found.</p>"

    return render_template(
        "index.html",
        table=table,
        graph=graph,
        industries=industries,
        experiences=experiences,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)