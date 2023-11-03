# Art Of Data
---
## Project Description
### Predicting the final sale price at fine art auctions
+++++++++++++++++\
\
**Overview:**\
With data retrieved from an art database create a model that would be capable of utilizing the numeric data to predict the price of 10 artists
sale prices. \
**Method**\
Regression / Classification / Deep Learning (Image Recognition).\
**Takeaways**\
TBD.\
**Skills:**\
Tensor Flow, Python, Natural Language Processing (NLP), Pandas, EDA, Facebook, SciKit-Learn, Classification

## Project Goals
---
- Accurately predicting an auction house estimate/ price for the art of 10 of the highest grossing artist 

## Initial Thoughts
--- 
- The artist will likely be the strongest drivers of the price point. The disparity of sale price for some artists is much higher meanwhile some will float within the same range.
- When incorporating image recognition the neural network will likely be able to recognize the works of the 10 artists, due to all of them being regarded for their individual styles.

## Planning
--- 
1. Acquire data from artnet.com
2. Data could only be retrieved in the form of a PDF
3. Clean aforementioned PDF/PDFs
3. Explore and analyze data for better insights on relative context to artist and their prices\
    a. Determine our baseline prediction\
    b. Are we beating the auction house's estimates?\
    c. Does logarithmically transforming hammer price allow for easier scaling for models?
    d. Stats test high frequency words to artists\ 
4. Start with Regression Model
5. Proceed to Classification
6. Document conclusions, recommendations, and next steps 
7. Move forward to MLP (Multilayer Percepitron) OR CNN (Convultional Neural Network)

## Data Dictionary
--- 
| Feature        | Definition                                   |
| ---            | ---                                          |
| artist  | The name of the artist who created the artwork |
| name | The name of the artwork |
| medium | The medium used to create the artwork |
| size | The size of the artwork |
| date_created | The date the artwork was created |
| lot | The lot number of the artwork in the auction |
| date_sold | The date the artwork was sold |
| auction_house | The name of the auction house where the artwork was sold |
| edition | The edition number of the artwork |
| estimated_price | The estimated price of the artwork before the auction (leakage) |
| hammer_price | The final price of the artwork at the auction |

## Reproducability Requirements
---
6. ### ⚠️ Disruption in reproducibility - [*artnet.com*](http://artnet.com)
***Terms of Service Ch.1***:\
    1. …*You may not modify, create derivative works from, participate in the transfer or sale of, post on the web, or in any way exploit the Site or Services or any portion thereof for any public or commercial use without the express written permission of artnet*…

## Conclusions 

## Recommendation
- Provide the art industry with a better tool that will potentially allow for higher liquidity within the market and provide more transparency and visibility for consumers