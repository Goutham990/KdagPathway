# Define a Pathway Table
papers_table = pw.io.read('google_drive_connector', format='csv', schema='Paper ID: int, content: str')

# Apply the model using Pathway
@pw.transform
def classify_and_recommend(papers_table):
    papers_table = papers_table.with_columns(
        publishable=papers_table['content'].apply(classify_paper),
        conference=papers_table['content'].apply(lambda content: recommend_conference(content)[0]),
        rationale=papers_table['content'].apply(lambda content: f"The paper aligns with {recommend_conference(content)[0]} focus areas.")
    )
    return papers_table

# Write results to Google Drive
pw.io.write(classify_and_recommend(papers_table), 'google_drive_connector', format='csv')