"""
Financial knowledge base for RAG.

This is the only data we ship in the repo. All other data (stock prices,
news, company info) is pulled live from yfinance and Tavily at runtime.

Each entry has a title, category, and detailed explanation. The Advisory
Agent uses semantic search over this corpus to answer financial education
questions.
"""


KNOWLEDGE_BASE = [
    {
        "title": "What is a 401(k)?",
        "category": "retirement",
        "content": (
            "A 401(k) is an employer sponsored retirement savings plan in the United States. "
            "Employees contribute a portion of their salary before taxes, which reduces their "
            "current taxable income. Many employers match a percentage of contributions, "
            "which is essentially free money. The funds grow tax deferred until withdrawal "
            "in retirement, typically after age 59 and a half. Annual contribution limits are "
            "set by the IRS and adjusted for inflation. A Roth 401(k) option allows post tax "
            "contributions with tax free withdrawals in retirement."
        ),
    },
    {
        "title": "What is an ETF?",
        "category": "investment-vehicles",
        "content": (
            "An Exchange Traded Fund (ETF) is a basket of securities that trades on a stock "
            "exchange like an individual stock. ETFs typically track an index, sector, "
            "commodity, or asset class. They offer diversification, lower fees than mutual "
            "funds, and intraday liquidity. Popular ETFs include SPY (S&P 500), QQQ (Nasdaq "
            "100), and VTI (total US market). ETFs are favored for their tax efficiency, "
            "transparency, and low expense ratios often below 0.1% annually."
        ),
    },
    {
        "title": "Dollar Cost Averaging explained",
        "category": "strategy",
        "content": (
            "Dollar cost averaging is an investment strategy where you invest a fixed amount "
            "of money at regular intervals regardless of market price. For example, investing "
            "500 dollars in an index fund every month. This approach reduces the impact of "
            "volatility by buying more shares when prices are low and fewer shares when "
            "prices are high. It removes emotional decision making and timing risk. Over long "
            "periods, this strategy tends to perform well and is the default approach for "
            "most 401(k) and retirement accounts."
        ),
    },
    {
        "title": "Understanding P/E Ratio",
        "category": "valuation",
        "content": (
            "The Price to Earnings (P/E) ratio compares a company's share price to its "
            "earnings per share. A P/E of 20 means investors pay 20 dollars for every 1 "
            "dollar of annual earnings. High P/E suggests investors expect strong future "
            "growth. Low P/E may indicate an undervalued stock or declining business. "
            "Compare P/E within the same industry since tech stocks typically trade higher "
            "than utilities. The S&P 500 historical average P/E is around 15 to 20."
        ),
    },
    {
        "title": "Roth IRA vs Traditional IRA",
        "category": "retirement",
        "content": (
            "Both are individual retirement accounts with annual contribution limits. "
            "Traditional IRA contributions are tax deductible now, and withdrawals in "
            "retirement are taxed as income. Roth IRA contributions are made with after tax "
            "money, but withdrawals in retirement are completely tax free including all "
            "growth. Roth is better if you expect to be in a higher tax bracket at "
            "retirement. Traditional is better if you need the tax break now. Roth has "
            "income limits that prevent high earners from contributing directly."
        ),
    },
    {
        "title": "Market Capitalization",
        "category": "valuation",
        "content": (
            "Market cap equals share price multiplied by total outstanding shares. It "
            "represents the total value of a company as perceived by the market. Large cap "
            "stocks (above 10 billion) are typically established companies like Apple and "
            "Microsoft. Mid cap stocks (2 to 10 billion) offer growth potential with "
            "moderate risk. Small cap stocks (below 2 billion) can grow faster but carry "
            "higher volatility. Market cap is used to weight index funds, so S&P 500 is "
            "more influenced by Apple than by a small company."
        ),
    },
    {
        "title": "Diversification and Portfolio Risk",
        "category": "strategy",
        "content": (
            "Diversification means spreading investments across different assets to reduce "
            "risk. A diversified portfolio might include US stocks, international stocks, "
            "bonds, real estate, and commodities. The idea is that different assets react "
            "differently to market conditions. When one falls, another may rise, smoothing "
            "overall returns. The classic 60/40 portfolio holds 60 percent stocks and 40 "
            "percent bonds. True diversification reduces unsystematic risk but cannot "
            "eliminate market wide systematic risk."
        ),
    },
    {
        "title": "Dividend Yield and Dividend Stocks",
        "category": "investment-vehicles",
        "content": (
            "Dividend yield is the annual dividend per share divided by the stock price. "
            "A stock at 100 dollars paying 4 dollars annually has a 4 percent yield. "
            "Dividend stocks provide regular income and are popular for retirees. Utilities, "
            "consumer staples, and real estate investment trusts (REITs) typically pay "
            "higher dividends. Dividend Aristocrats are S&P 500 companies that have "
            "increased dividends for 25 consecutive years. Reinvested dividends have "
            "historically contributed a significant portion of total stock returns."
        ),
    },
    {
        "title": "Bull vs Bear Market",
        "category": "market-concepts",
        "content": (
            "A bull market is a period of rising stock prices, typically defined as a 20 "
            "percent increase from recent lows. Bull markets are characterized by investor "
            "optimism, economic growth, and strong corporate earnings. A bear market is the "
            "opposite: a 20 percent decline from recent highs, driven by pessimism, "
            "recession fears, or market shocks. The average bull market lasts around 6 to 7 "
            "years and gains over 150 percent. Average bear markets last about 1 year with "
            "declines around 30 percent."
        ),
    },
    {
        "title": "Index Funds explained",
        "category": "investment-vehicles",
        "content": (
            "An index fund is a mutual fund or ETF that tracks a specific market index like "
            "the S&P 500 or total stock market. Instead of trying to beat the market, index "
            "funds aim to match market performance. They have very low expense ratios often "
            "below 0.05 percent. Warren Buffett recommends index funds for most investors "
            "because over long periods they outperform most actively managed funds. Popular "
            "choices include VOO (Vanguard S&P 500), VTI (total US market), and VXUS "
            "(international stocks)."
        ),
    },
    {
        "title": "Compound Interest and Time Value",
        "category": "fundamentals",
        "content": (
            "Compound interest is interest earned on both the original principal and "
            "accumulated interest. Over time this creates exponential growth. Investing "
            "10,000 dollars at 7 percent annual return grows to about 76,000 in 30 years "
            "without adding anything. Starting early is the most powerful wealth building "
            "lever. Someone investing 5,000 annually from age 25 to 35 (10 years, 50,000 "
            "total) typically ends up with more at 65 than someone investing 5,000 annually "
            "from 35 to 65 (30 years, 150,000 total) due to compound growth."
        ),
    },
    {
        "title": "Bonds and Fixed Income",
        "category": "investment-vehicles",
        "content": (
            "A bond is a loan from you to a government or corporation. The issuer pays "
            "periodic interest (coupon payments) and returns the principal at maturity. "
            "US Treasury bonds are considered the safest investments in the world. "
            "Corporate bonds pay higher yields but carry default risk. Bond prices move "
            "inversely to interest rates: when rates rise, existing bond prices fall. Bonds "
            "are typically held for stability, income, and to offset stock market "
            "volatility. A common allocation is 100 minus your age in stocks, rest in bonds."
        ),
    },
    {
        "title": "Portfolio Rebalancing",
        "category": "strategy",
        "content": (
            "Rebalancing is the process of restoring your portfolio to target asset "
            "allocations. If you target 60 percent stocks and 40 percent bonds but stocks "
            "rally to 70 percent, you sell stocks and buy bonds to return to 60/40. "
            "Rebalancing forces you to buy low and sell high systematically. It can be done "
            "quarterly, annually, or when allocations drift beyond a threshold like 5 "
            "percent. Target date funds rebalance automatically. Rebalancing maintains your "
            "risk profile and prevents any single asset from dominating your portfolio."
        ),
    },
    {
        "title": "Capital Gains Tax",
        "category": "taxes",
        "content": (
            "When you sell an investment for more than you paid, the profit is a capital "
            "gain subject to tax. Short term capital gains (held less than one year) are "
            "taxed as ordinary income. Long term capital gains (held more than one year) "
            "are taxed at preferential rates of 0, 15, or 20 percent depending on income. "
            "Tax loss harvesting means selling losers to offset gains and reduce tax "
            "liability. Retirement accounts like 401(k) and IRA defer or eliminate capital "
            "gains taxes, which is why they are powerful wealth building tools."
        ),
    },
    {
        "title": "Risk Tolerance and Asset Allocation",
        "category": "strategy",
        "content": (
            "Risk tolerance is your ability and willingness to endure portfolio volatility. "
            "It depends on age, income stability, goals, and psychology. Younger investors "
            "typically have higher risk tolerance because they have more time to recover "
            "from downturns. A 25 year old might hold 90 percent stocks, while a 60 year "
            "old might hold 50 percent stocks and 50 percent bonds. Aggressive allocations "
            "offer higher expected returns but larger drawdowns. Conservative allocations "
            "reduce volatility but also limit long term growth potential."
        ),
    },
    {
        "title": "Emergency Fund basics",
        "category": "fundamentals",
        "content": (
            "An emergency fund is money set aside for unexpected expenses like medical "
            "bills, car repairs, or job loss. The standard recommendation is 3 to 6 months "
            "of essential expenses in a high yield savings account. Emergency funds should "
            "be liquid and stable, not invested in stocks or bonds. Having an emergency "
            "fund prevents you from selling investments at a bad time or taking on high "
            "interest debt. It is the foundation of any financial plan and should be built "
            "before aggressive investing."
        ),
    },
]
