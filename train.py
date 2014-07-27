import math
import operator
import time
import sys
from collections import defaultdict
from collections import Counter
#Implementation of Naive Bayes Classifier that tarins and then classifies the test data into two classes given the trained data.
__author__ = "Ozge Ozbek, ozgeozbek@gmail.com"
class NB:
	def __init__(self, trainedfile):
		self.dictionary=[] #holds all words in the training set
		self.classvocabulary=defaultdict(list) #holds all words in the training set divided into classes
		self.stopwords=['a', 'an', 'and', 'are', 'at', 'be', 'by', 'for', 'from', 'has', 'in', 'is','its', 'it','of','on','that','the', 'to','was','were','with','will','']
		self.documentclasses=['0','1'] #holds all available classes in the training set
		self.traindata=self.gettraindata(trainedfile) # initializes all above structures related to training set
		self.wordcounts={}
		self.setcounttokensofterm()
		self.totalwordsinvocab=self.documentcount()
		self.listofpriors=self.setprior(self.documentclasses)
		self.constantB=len(self.dictionary)

#Train: Load data
	def gettraindata(self, trainedfile): 
	#this function loads the clean final as a dictionary
		dict = {}
		counter=0
		with open(trainedfile) as f:
		#with open("music_dump_utf8_final.txt") as f:
		#with open("testin.txt") as f:
			for line in f:
				parts = line.split('\t')
				key= parts[0]
				classification= parts[1]
				doctemp = parts[2]
				doc=doctemp[:-1]
				doc= doc.split(' ')
				doc=self.removestopwords(doc)
				dict[key]=(classification,doc)
				self.concatenatetextofalldocsinclass(doc, classification)
				self.update_progress(counter/90000.0) #ignore this, it is used to create a progress bar on command line
				counter+=1
				for term in doc:
					if term.lower() not in self.dictionary:
						self.dictionary.append(term.lower()) #this holds all the words in the dictionary
		f.close()
		return dict
	
	def isstopword(self, term):
		isstop= term.lower() in self.stopwords
		return isstop
		
	def removestopwords(self, listoftokens):
		temptokenlist=[]
		for term in listoftokens:
			if(self.isstopword(term)==False):
				temptokenlist.append(term)
		return temptokenlist
	#concatenate all terms in all documents that belong to that class
	#called text_ct==> this is the single document that contains all documents belonging to the class
	def concatenatetextofalldocsinclass(self, doc,classification):		
		for term in doc:
			self.classvocabulary[classification].append(term.lower())	

#Train: 		
	def setcounttokensofterm(self):
		for cls in self.classvocabulary:
			counts=Counter(self.classvocabulary[cls])
			self.wordcounts[cls]=counts

#Train: Find prior probability
	#P(c) is the prior probability of a document occurring in class c (c=1 or c=0)
	#N_c/N where N is total number of documents				
	def setprior(self, docclass):
		priors=[]
		for cls in docclass:
			priors.append(self.priorcount(cls)/self.totalwordsinvocab)
		return priors
			
	def getprior(self,docclass):
		return self.listofpriors[int(docclass)] 	
	#This returns N_c: The number of documents only belonging to the class we want
	def priorcount(self, docclass):
		priorscount=0
		ks=list(self.traindata.keys())
		for key in ks:
			if(self.traindata[key][0]==docclass):
				priorscount=priorscount+1
		return priorscount
					
	#This returns 'N': the number of all documents in the input traindata
	def documentcount(self):
		docscount=len(self.traindata.keys())
		return docscount
#Train:		
	#function below counts the number of occurrences of a given term in the set of all tokens
	#that we found to be belonging to a class in the training set.	
	
	def counttokensofterm(self,term, cls):
		return self.wordcounts[cls][term]	

	#below func calculates the conditional probability of a given term and a class
	#class is either '1' or '0' 
	def condprob(self,term, docclass):
		#find the numerator first: how many times this term occurs in given class, and add 1
		tct=self.counttokensofterm(term, docclass)
		tct=tct+1
		#find the denominator now: ie total length of the vocabulary in the class plus B constant
		sum_text_ct=len(self.classvocabulary[docclass])
		result=tct/(sum_text_ct+self.constantB) #constantB is the total size of the training set, unique words
		return result
	
	#this function takes the test document and returns the argmax class for it
	def applymultinomialnb(self, document):
		#for all the terms in the document we have to find log(P(c)*P(word/c))
		scores= {}
		for i in self.documentclasses:
			#score=math.log(self.getprior(i))
			score=self.getprior(i)
			for term in document:
				#score+=math.log(self.condprob(term,i))
				score*=self.condprob(term.lower(),i)
			scores[i]=score
		#now find which class has higher score
		argmax=max(scores.items(), key=operator.itemgetter(1))[0]
		return argmax
	
	def testandwriteclass(self, testfile, classifiedfile ):
		#counter=1
		#clean test file, then open
		outfile = open(classifiedfile,'w')
		with open(testfile) as f:
		#with open("music_dump_unknown_10000_utf8_final.txt") as f:
		#with open("test.txt") as f:
			for line in f:
				parts = line.split('\t')
				key= parts[0]
				doctemp = parts[2]
				doc=doctemp[:-1]
				doc= doc.split(' ')
				doc=self.removestopwords(doc)
				classification=self.applymultinomialnb(doc) #calculate its score and find the best class
				outline=key+'\t'+classification+'\t'+doctemp
				outfile.write(outline)
		f.close()
		outfile.close()
##############################################################################
	def precisionandrecall(self, expertmarkedfile, classfierfile):
		guessdict={}
		realdict={}
		with open(expertmarkedfile) as f:
			for line in f:
				parts = line.split('\t')
				key= parts[0]
				classification= parts[1]
				realdict[key]=classification
		f.close()
		with open(classfierfile) as t:# get the guessed values
			for line in t:
				parts = line.split('\t')
				key= parts[0]
				classification= parts[1]
				guessdict[key]=classification
		#calculate precision: how many predictions are correct:
		reallist=list(realdict.keys())
		guesslist=list(guessdict.keys())
		truepositive=0
		totalguessedpositive=0
		for key in guesslist:
			if(guessdict[key][0]=='1'):
				totalguessedpositive+=1
				if(guessdict[key][0]==realdict[key][0]):
					truepositive+=1
		print('total # of 1s marked by classifier: ',totalguessedpositive)
		print('number of matching 1s in the prediction: ',truepositive)
		print('precision is: ', truepositive/totalguessedpositive)
		
		#calculate recall: percentage of all guesses of "1" within all expert-marks of "1"
		totalmarkedpositive=0
		for key in reallist:
			if(realdict[key][0]=='1'):
				totalmarkedpositive+=1

		print('total # of 1s marked by experts: ',totalmarkedpositive)
		print('number of matching 1s in the prediction: ',truepositive)
		print('recall is: ', truepositive/totalmarkedpositive)
##############################################################################
#this is not related to the assignment, it only shows a progress bar on screen as it takes around 7 mins for the training document to be loaded
	def update_progress(self,progress):
		barLength = 70 # Modify this to change the length of the progress bar
		status = ""
		if isinstance(progress, int):
			progress = float(progress)
		if not isinstance(progress, float):
			progress = 0
			status = "error: progress var must be float\r\n"
		if progress < 0:
			progress = 0
			status = "Halt...\r\n"
		if progress >= 1:
			progress = 1
			status = "Done...\r\n"
		block = int(round(barLength*progress))
		text = "\rDocument scan completion: [{0}] {1}% {2}".format( "="*block + "-"*(barLength-block), format(progress*100,'.2f').rstrip('0').rstrip('.'), status)
		sys.stdout.write(text)
		sys.stdout.flush()

m=NB("big_pool_trained.txt")
k=m.testandwriteclass("small_pool_trained.txt",'outResult_22Feb-3.txt')
m.precisionandrecall("small_pool_trained.txt","outResult_22Feb-3.txt")
