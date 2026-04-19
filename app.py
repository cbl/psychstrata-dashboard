from dash import Dash, html

from api import register_api
from callbacks import register_callbacks
from components import (
    create_cif_card,
    create_feature_form,
    create_header_card,
    create_llm_summary_card,
    create_prediction_card,
    create_shap_card,
    create_tsne_card,
)
from config import CARD
from models import load_model
from models.random_forest_mdd.features import CATEGORY_ORDER


model = load_model()

app = Dash(__name__)
app.title = f"Treatment Resistance Classifier ({model.diagnosis})"
server = app.server


def create_layout() -> html.Div:
    form = create_feature_form(model.features, CATEGORY_ORDER)
    left_column = html.Div(
        [
            html.Div(
                [
                    html.H4("Enter patient features",
                            style={"margin": "0 0 8px 0", "fontSize": "18px"}),
                    form,
                ],
                style=CARD,
            ),
            create_llm_summary_card(),
        ],
        style={"flex": "1.7", "minWidth": "320px", "display": "flex",
               "flexDirection": "column", "gap": "12px"},
    )
    right_column = html.Div(
        [create_prediction_card(), create_cif_card(), create_shap_card(), create_tsne_card()],
        style={"flex": "1", "minWidth": "360px", "display": "flex",
               "flexDirection": "column", "gap": "12px"},
    )
    main_content = html.Div(
        [left_column, right_column],
        style={"display": "flex", "gap": "20px", "marginTop": "20px",
               "flexWrap": "wrap", "alignItems": "stretch"},
    )
    footer = html.Div(
        "This demo is for educational purposes only and is not a medical device.",
        style={"fontSize": "12px", "color": "#6b7280", "textAlign": "center", "marginTop": "16px"},
    )
    return html.Div(
        [
            html.Div(
                [create_header_card(model.auc, model.diagnosis, model.name),
                 main_content, footer],
                style={"maxWidth": "1280px", "margin": "0 auto"},
            ),
        ],
        style={"padding": "24px", "backgroundColor": "#f6f7fb", "fontFamily": "Inter, Arial, sans-serif"},
    )


app.layout = create_layout()
register_api(server, model)
register_callbacks(app, model)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
