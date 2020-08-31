'''
Used to check whether the client has received enough matching reply messages
'''

class ReplyTracker:
    def __init__(self,Response,ResponseCt,Done,ciphertext,result,n):
        self.Response = Response
        self.SentRequest = {}
        self.ResponseCt = ResponseCt
        self.Done = Done
        self.ciphertext = ciphertext
        self.n = n
        self.ReplyMap = {}
    
    '''
    Used to verify whether the client has ever sent a request
    @param t, timestamp
    '''
    def initiateCt(self,t):
        self.SentRequest[t] = True

    '''
    Check whether the client has sent a request
    @param, t, timestamp
    '''
    def checkIndex(self,t):
        try:
            return self.SentRequest[t]
        except:
            return False

    '''
    increase the counter by one of the reply matches previous replies
    @param i, string, serialized reply (v, t, c, r)
    '''
    def increaseCt(self,i):
        try:
            self.ResponseCt[i]+=1
        except:
            self.ResponseCt[i]=1
            self.Done[i] = 0

    '''
    Check whether enough replies have been received
    @param i, index for verifying the message
    '''
    def complete(self,i):
        try:
            return self.Done[i]
        except:
            self.Done[i] = 0
            return self.Done[i]
    '''
    Append response
    '''
    def appendNew(self,i,content):
        try:
            self.Response[i].append(content)
        except:
            self.Response[i] = []
            self.Response[i].append(content)
    
    '''
    Insert ciphertext to in memory table
    '''
    def putcipherText(self,i,ciphertext):
        self.ciphertext[i] = ciphertext
