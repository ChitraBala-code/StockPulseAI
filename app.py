
import os

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="StockPulse AI",
    page_icon="📈",
    layout="wide",
)

# NSE symbols with the Yahoo Finance suffix.
STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Larsen & Toubro": "LT.NS",
    "ITC": "ITC.NS",
    "State Bank of India": "SBIN.NS",
    "Tata Motors Passenger Vehicles": "TMPV.NS",
}

st.title("📈 StockPulse AI")
st.caption(
    "Indian share-market updates | Daily prices | AI-assisted insights"
)

st.info(
    "Data is provided through Yahoo Finance via yfinance. "
    "Quotes may be delayed, incomplete, or unavailable. "
    "Verify important figures with your broker or an official source."
)


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data():
    rows = []
    errors = []

    for company, ticker in STOCKS.items():
        try:
            history = yf.Ticker(ticker).history(
                period="5d",
                interval="1d",
                auto_adjust=False,
            )

            if history.empty or len(history) < 2:
                errors.append(company)
                continue

            # Use the two latest available trading sessions.
            previous = float(history["Close"].iloc[-2])
            latest = float(history["Close"].iloc[-1])

            if previous <= 0:
                errors.append(company)
                continue

            change = latest - previous
            change_pct = (change / previous) * 100

            rows.append({
                "Company": company,
                "Ticker": ticker,
                "Latest Close (₹)": round(latest, 2),
                "Previous Close (₹)": round(previous, 2),
                "Change (₹)": round(change, 2),
                "Change (%)": round(change_pct, 2),
                "Volume": int(history["Volume"].iloc[-1]),
                "Session Date": str(
                    history.index[-1].date()
                ),
            })

        except Exception:
            errors.append(company)

    return pd.DataFrame(rows), errors


with st.sidebar:
    st.header("Settings")
    st.write("Market: India")
    st.write("Exchange focus: NSE")
    st.write("Report: Daily")
    if st.button("Refresh market data"):
        get_stock_data.clear()
        st.rerun()

with st.spinner("Fetching the latest available daily prices..."):
    stocks, errors = get_stock_data()

if stocks.empty:
    st.error(
        "No stock data could be retrieved. Check your internet "
        "connection and try again later."
    )
    st.stop()

stocks = stocks.sort_values("Change (%)", ascending=False)

# Dashboard metrics
advancers = int((stocks["Change (%)"] > 0).sum())
decliners = int((stocks["Change (%)"] < 0).sum())
average_change = float(stocks["Change (%)"].mean())

c1, c2, c3 = st.columns(3)
c1.metric("Shares Rising", advancers)
c2.metric("Shares Falling", decliners)
c3.metric("Average Change", f"{average_change:+.2f}%")

st.subheader("Leading Shares")


def color_change(value):
    if value > 0:
        return "color: green"
    if value < 0:
        return "color: red"
    return ""


styled = stocks.style.map(
    color_change,
    subset=["Change (₹)", "Change (%)"],
)

st.dataframe(
    styled,
    width=True,
    hide_index=True,
)

st.subheader("Daily Price Changes")

chart_data = stocks.set_index("Company")[["Change (%)"]]
st.bar_chart(chart_data)

# Simple grounded summary: this works even before AI is configured.
st.subheader("Market Summary")

best = stocks.iloc[0]
worst = stocks.iloc[-1]

summary = (
    f"Among the {len(stocks)} tracked shares, "
    f"{advancers} rose and {decliners} fell in the latest "
    f"available sessions. The average percentage change "
    f"was {average_change:+.2f}%. "
    f"{best['Company']} had the strongest change "
    f"({best['Change (%)']:+.2f}%), while "
    f"{worst['Company']} had the weakest change "
    f"({worst['Change (%)']:+.2f}%)."
)

st.write(summary)

# Optional GenAI summary using a Hugging Face token.


def get_hf_token():
    token = os.getenv("HF_TOKEN", "")

    if token:
        return token

    try:
        return st.secrets.get("HF_TOKEN", "")
    except Exception:
        return ""


if st.button("Generate AI Market Explanation"):
    token = get_hf_token()

    if not token:
        st.warning(
            "The data dashboard works, but AI is not configured yet. "
            "Next, add your Hugging Face token to enable AI summaries."
        )
    else:
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(
                model="Qwen/Qwen2.5-7B-Instruct",
                token=token,
            )

            market_data = stocks[
                ["Company", "Latest Close (₹)", "Change (%)"]
            ].to_dict(orient="records")

            response = client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a cautious Indian stock-market "
                            "information assistant. Explain only what "
                            "the supplied data supports. Do not invent "
                            "news or causes. State that daily price "
                            "changes alone cannot establish why a stock "
                            "moved. Do not give personalized buy or "
                            "sell advice."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Explain this daily stock snapshot in "
                            "simple English. Mention the strongest "
                            "and weakest shares and the overall pattern. "
                            "Clearly separate observations from possible "
                            "interpretations. Data: "
                            + str(market_data)
                        ),
                    },
                ],
                max_tokens=300,
            )

            st.markdown(response.choices[0].message.content)

        except Exception as exc:
            st.error(
                "The AI service could not generate a summary. "
                "Check your Hugging Face token, model access, "
                "internet connection, and inference availability. "
                f"Technical detail: {exc}"
            )

if errors:
    st.warning(
        "Data could not be retrieved for: "
        + ", ".join(errors)
        + ". These shares are excluded from the dashboard."
    )

st.caption(
    "Educational information only. Not investment advice. "
    "Market data may be delayed or inaccurate."
)
