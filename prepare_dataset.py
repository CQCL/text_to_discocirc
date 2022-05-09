###############################################################################
# This script extracts data for the first 3 bAbI tasks from the relevant .txt
# files.
###############################################################################

# read the file
with open('tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt') as f:
    lines = f.readlines()


# split the lines into stories
# record the first line location of new stories
story_splits = [i for i, line in enumerate(lines) if line[0:2] == '1 ']
# have no more need for line indices - delete these
lines = [' '.join(line.split(' ')[1:]) for line in lines]
# also delete . and \n
lines = [line.replace('.', '').replace('\n','') for line in lines]
stories = [lines[i:j] for i, j in zip(story_splits, story_splits[1:]+[None])]

# create context and QnA pairs
contexts = []
qnas = []
for story in stories:
    # record the lines in the story corresponding to questions
    question_splits = [i for i, line in enumerate(story) if '?' in line]
    for index in question_splits:
        # record the context corresponding to each question
        contexts.append([line for line in story[:index] if '?' not in line])
        # record the question
        qnas.append(story[index])


# split qna into questions and answers
questions = [qna.split(' \t')[0] for qna in qnas]
answers = [qna.split('\t')[1] for qna in qnas]

# convert question answers to statements
answer_statements = []

for question, answer_word in zip(questions, answers):
    question_word = question.split()[-1][:-1]
    answer_statements.append('The '+question_word+' is in the '+answer_word)
