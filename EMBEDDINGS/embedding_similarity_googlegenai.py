from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

documents=[
"Sachin Tendulkar, often referred to as the God of Cricket, had a career spanning 24 years during which he set the unprecedented record of scoring 100 international centuries. His 34,357 international runs remain the highest in the sport's history, and his pivotal performance in India's 2011 World Cup victory cemented his status as a national icon.",
"Sir Don Bradman is widely considered the greatest batsman to have played the game, famously holding a Test batting average of 99.94. His dominance in the 1930s was so absolute that his statistical benchmark remains untouched by any other player, making him a mythical figure in the history of the sport.",
"Virat Kohli has established himself as one of the most prolific run-scorers of the modern era, holding numerous records including the most runs in T20 Internationals. Known for his intense fitness and aggressive batting style, he has been a consistent match-winner for India across all formats for over a decade.",
"Sir Garfield Sobers is regarded as arguably the greatest all-rounder in the history of cricket. Capable of batting, bowling various styles of spin and seam, and fielding brilliantly, his versatility and longevity defined the standard for the modern all-rounder role.",
"Ricky Ponting was a formidable Australian batsman and one of the most successful captains in cricket history. Under his leadership, Australia dominated the international scene, and his personal batting record—highlighted by over 27,000 international runs—places him among the elite players of the 21st century.",
"Jacques Kallis of South Africa is statistically one of the most complete cricketers to ever play, amassing over 25,000 international runs and taking more than 500 wickets. His ability to anchor an innings with the bat while serving as a reliable fast-medium bowler made him an invaluable asset to his team for nearly 15 years.",
"Muttiah Muralitharan of Sri Lanka is the highest wicket-taker in the history of Test cricket, famous for his unique off-spin action and incredible control. His ability to spin the ball on almost any surface allowed him to dominate opposition batting lineups and play a critical role in Sri Lanka’s rise as a cricketing power.",
"MS Dhoni is celebrated as one of India's most successful captains, leading the team to victory in the 2011 World Cup and the 2007 T20 World Cup. Beyond his leadership, his rapid wicketkeeping and calm demeanor as a finisher under pressure have made him one of the most beloved figures in global cricket.",
"Brian Lara of the West Indies was a master of big scores, including the record for the highest individual score in a Test match with an unbeaten 400. His elegant stroke play and ability to single-handedly dismantle world-class bowling attacks ensured his legacy as one of the most gifted left-handed batters in history",
"Adam Gilchrist revolutionized the role of the wicketkeeper-batsman by proving that a keeper could also be a top-order match-winner. His aggressive batting style and ability to score quickly changed how teams structured their middle order, earning him recognition as one of the most influential cricketers of the 21st century"
]

embeddings=GoogleGenerativeAIEmbeddings(model='gemini-embedding-2',dimensions=300)
query="all rounder"
doc_embeddings=embeddings.embed_documents(documents)
query_embedding=embeddings.embed_query(query)
scores=cosine_similarity([query_embedding],doc_embeddings)[0]
result=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(f"The most revelant result is:{documents[result[0]]}")
print(f"The Similarity Score is {result[1]}")