import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os


# 	Funcion que compara todos los features de referencia con
#	el feature pasado como parametro y devuelve la clase a la
#	que pertenece.

def takeSecond(elem):
    return elem[1]

def comparator(featureToCompare,color):
	files=[]
	ranking=[]
	# obtain names of reference features

	#features vgg16
	pathFeatures= "./refRobot/"+featureToCompare[1]+"/"+color+"/"
	#features resnet50
	#pathFeatures= "./imgRef/"+featureToCompare[1]+"/"
	for r, d, f in os.walk(pathFeatures):
		for file in f:
			if '.npy' in file:
				files.append(os.path.join(r, file))		

	maximo=[0,0]
	#print(files)

	# obtain simility value between reference features
	for f in files:
		reference= np.load(f)

		# suma guarda valor temporal. Maximo guarda valor maximo y clase
		
		cos=0
		#for i in range(7):
			#suma += np.linalg.norm(cosine_similarity(featureToCompare[0][i],reference[0][i]))

		a=featureToCompare[0][0].flatten('F')
		b= reference[0].flatten('F')
		dot = np.dot(a, b)
		norma = np.linalg.norm(a)
		normb = np.linalg.norm(b)
		cos = dot / (norma * normb)
	

		# obtengo la maxima similitud
		#if(cos>maximo[0]):
		#	maximo[0] = cos
		#	maximo[1] = int(f[len(pathFeatures):][:1])
		
		
		ranking.append([f[f.rfind("/")+1:f.find("-")],cos])
	ranking.sort(key=takeSecond,reverse=True)
	return ranking