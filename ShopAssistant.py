class ShopAssistant:

    def __init__(self, name):
        self.name = name
        self.reply = ""
        self.prep = {"on", "to", "the", "over", "under", "behind", "forward", "at",
                     "in", "above", "below", "in front of","in the front of", "besides"}
        self.be = {"am", "is", "are", "'s", "'re", "was", "were"}
        self.mall_map = {"shoes": "first floor", "cloth": 'second floor', "jewelry": 'first floor',
                         "furniture": "forth floor", "restroom": "second floor",
                         "restaurant": "lower floor", "food": "lower floor",
                         "drink": "lower floor"}
        self.end_flag = False
        self.complain_flag = False

    def assistant(self, input):

        input = input.lower()

        #guide
        if input.find("where") != -1 and input.find("such") == -1:
            input = input.replace("where", "")
            for word in self.prep:
                input = input.replace(word, '')
            for word in self.be:
                input = input.replace(word, '')

            #find place in map
            find_flag = False
            input_list = input.split()
            for k in input_list:
                if k in self.mall_map.keys():
                    find_flag = True
                    self.reply = input + " is on the " + self.mall_map.get(k)
                if find_flag:
                    break
            #judgement
            tmp = "I didn't find" + input
            self.reply = self.reply if find_flag else tmp
            self.complain_flag = False

        # end the conversation
        elif input.find("bye") != -1 or input.find("thank you") != -1 or input.find("thanks") != -1:
            self.end_flag = True
            self.reply = "Bye! Welcome again!"
        elif self.complain_flag:
            self.reply = self.deal_complain(input)
            self.complain_flag = False
        else:
            self.reply = "Thanks for your comments! Any other thing I can help?"
            self.complain_flag = False

        return self.reply

    def deal_complain(self,input):
        #add something to deal with input
        val = "I am sorry to hear that! I will record this. " + \
                    "If you need further help, please leave your email or phone number. " + \
                    "If you would like to call a real agent, please press the button."
        return val

    def intro(self):
        return "Hello! This is " + self.name + ", your shop assistant. Please use word where," +\
        "and let me found the place for you. Use bye to end the conversation."

    def get_end_flag(self):
        return self.end_flag

    def bye(self):
        return "Thank you for visiting! Have a wonderful day!"

    def set_complain(self):
        self.complain_flag = True
        return