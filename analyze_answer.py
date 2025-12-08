import pandas as pd

df = pd.read_csv('output/LLM_answers.csv')

llm_answer_cols = df.columns[3:]
matches_df = df[llm_answer_cols].eq(df['correct_answer'], axis=0)
match_count = matches_df.astype(int).sum(axis=1)

df['match_count'] = match_count
df['majority_vote'] = df[llm_answer_cols].mode(axis=1)[0]
df['success_rate'] = match_count / len(llm_answer_cols)

# calc stat
overall_successful_rate = sum(df['success_rate'])/len(df['success_rate'])
print(f'overall_successful_rate: {overall_successful_rate}')

# get Q&A that LLM sometimes gets correct
print(df)