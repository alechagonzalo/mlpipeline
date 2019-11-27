import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os


# 	Funcion que compara todos los features de referencia con
#	el feature pasado como parametro y devuelve la clase a la
#	que pertenece.

def takeSecond(elem):
    return elem[1]

def comparator(featureToCompare,color,aspect):
	files=[]
	ranking=[]
	# obtain names of reference features


	# si existe filtro de color, comparar con todas las features
	if color !=False:
		pathFeatures= "./refRobot/"+featureToCompare[1]+"/"+str(color)+"/"
		for r, d, f in os.walk(pathFeatures):
			for file in f:
				if '.npy' in file:
					files.append(os.path.join(r, file))		

	else:
		pathFeatures= "./refRobot/"+featureToCompare[1]+"2/"
		for r, d, f in os.walk(pathFeatures):
			for file in f:
				if '.npy' in file:
					files.append(os.path.join(r, file))	

	if len(files)==0:
		return False
		
	maximo=[0,0]

	#print(files)

	# obtain simility value between reference features
	for f in files:
		reference= np.load(f)
		# obtengo la relacion de aspecto de la feature
		rel=f[f.rfind("-")+1:f.rfind(".")]
		
		cos=0

		a=featureToCompare[0][0].flatten('F')
		b= reference[0].flatten('F')
		dot = np.dot(a, b)
		norma = np.linalg.norm(a)
		normb = np.linalg.norm(b)
		cos = dot / (norma * normb)
	

		# guardo el nombre de la feature, su similitud y su relacion de aspecto
		ranking.append([f[f.rfind("/")+1:f.find("-")],cos,float(rel)])

	#ordeno el ranking de acuerdo a la similitud	
	ranking.sort(key=takeSecond,reverse=True)

	return ranking