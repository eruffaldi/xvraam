# xvraam.py new gendom save export "list objs" "load out1.aam" "list objs" save quit
#Changes: preparation for multitexturing, first texturename
__author__ = "Emanuele Ruffaldi"
__url__ = ["Teslacore, http://www.teslacore.it"]
__version__ = "0.09.20101109"
MAXWEIGHTS = 100

from collections import defaultdict
import re,os
import zipfile,cStringIO,math,netvars,array,struct,cPickle
import xvr.knn as knn
import math,numpy
import json

def sortandshorten(bws,maxn):
	bws.sort(key=lambda x: x[1],reverse=True)
	if len(bws) > maxn:
		# sort by decreasing weight, then cut array
		# compute the sum of all new weights                             
		bws = bws[0:maxn]
		nw = 1.0/sum([bw[1] for bw in bws]) # now the sum is less than 1
		bws = [(bw[0],bw[1]*nw) for bw in bws]
	return bws
	
class PointVec3(knn.Point):
	def __init__(self,data,idx):
		self.data = data
		self.baseIndex = idx
	def __repr__(self):
		return str("point %d(%f %f %f)" % (self.baseIndex,self.data[0],self.data[1],self.data[2]))
class TriangleVertex:
	def __init__(self,v,t=None,s=0):
		self.v = v
		if t is None:
			self.t = []
		else:
			self.t = t
		self.n = None
		self.s = s
	def asTuple(self): # UGLY
		return (self.s,self.v,)+ tuple(self.t)
def objtriscan(x):
	q = (x+"//").split("/")
	a = q[0]
	b = q[1]
	c = q[2]
	if c != "" and c != a:
		raise "Unsupported normal different from vertex"
	if b == "":
		return TriangleVertex(int(a)-1)
	else:
		return TriangleVertex(int(a)-1,(int(b)-1,))
class SimpleOBJLoad:
	def __init__(self,name):
		self.normals = [];
		self.vertices = [];
		self.tris = [];
		self.texcoords = [];
		q = open(name,"rb")
		for x in q:
			x = x.strip()
			if len(x) == 0 or x[0] == "#":
				continue
			if x.startswith("vn "):
				self.normals.append([float(y) for y in x[3:].split(" ")])
			elif x.startswith("vt "):
				self.texcoords.append([float(y) for y in x[3:].split(" ")])
			elif x.startswith("v "):
				self.vertices.append([float(y) for y in x[2:].split(" ")])
			elif x.startswith("f "):
				# TODO not implemented negative
				q = [objtriscan(y) for y in x[2:].split(" ")]
				self.tris.append(q)
		self.maxtexcoords = len(self.texcoords) != 0 and 1 or 0
	def __repr__(self):
		return str(("objmesh",len(self.vertices),len(self.texcoords),len(self.normals),len(self.tris)))
class Stats:
    def __init__(self):
        self.mean = None
        self.min = None
        self.max = None
        self.count = 0
    def push(self,x):
        if self.count == 0:
            self.mean = x
            self.min = x
            self.max = x
            self.count = 1
        else:
            self.count = self.count + 1
            self.mean += x
            self.max = max(x,self.max)
            self.min = min(x,self.min)
    def __str__(self):
        return "count:%d min:%f max:%f mean:%f span:%f" % (self.count,self.min,self.max,self.mean/self.count,self.max-self.min)
def asvector(x):
    return array.array("f",x);
def intvectorasstr(x):
    return struct.pack("%di" % len(x),*x)
def buildTriangle(vtxs,texs,s=0):
	"""Each Triangle Vertex contains all indices plus the smoothing group"""
	r = []
	for i in range(0,3):
		r.append(TriangleVertex(vtxs[i],[texs[j][i] for j in range(0,len(texs))],s))
	return r
def vadd(a,b):
    return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]
def vsub(a,b):
    return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]
def vscale(a,s):
    return [a[0]*s,a[1]*s,a[2]*s]
def vcross(a,b):
    return [a[1]*b[2]-b[1]*a[2],a[2]*b[0]-b[2]*a[0],a[0]*b[1]-b[0]*a[1]]
def vlength(a):
    return math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
def vnormalize(a):
    l = vlength(a)    
    return l != 0 and vscale(a,1.0/vlength(a)) or a
def genskydom(radius=1000,hori_res=30,vert_res=8,half_sphere_fraction=1.1,image_percentage = 1.0):
	azimuth_step = 2.*math.pi/hori_res;
	elevation_step = half_sphere_fraction*math.pi/2./vert_res;
	NumOfVertices = (hori_res+1)*(vert_res+1);
	NumOfFaces = (2*vert_res-1)*hori_res;
	vertices = []
	tcoords = []
	indices = []
	azimuth = 0
	for k in range(0,hori_res+1):
		elevation = math.pi/2
		for j in range(0,vert_res+1):
			vertices.append((radius*math.cos(elevation)*math.sin(azimuth),radius*math.cos(elevation)*math.cos(azimuth),radius*math.sin(elevation)))
			tcoords.append((k/float(hori_res+1),1-j/float((vert_res+1)*image_percentage)))
			elevation -= elevation_step
		azimuth += azimuth_step
	for k in range(0,hori_res):
		#indices.append((vert_res+2+(vert_res+1)*k,1+(vert_res+1)*k,0+(vert_res+1)*k));
		indices.append((vert_res+2+(vert_res+1)*k,0+(vert_res+1)*k,1+(vert_res+1)*k));
		for j in range(1,vert_res):
			#indices.append((vert_res+2+(vert_res+1)*k+j,1+(vert_res+1)*k+j,0+(vert_res+1)*k+j));
			#indices.append((vert_res+1+(vert_res+1)*k+j,vert_res+2+(vert_res+1)*k+j,0+(vert_res+1)*k+j));
			indices.append((vert_res+2+(vert_res+1)*k+j,0+(vert_res+1)*k+j,1+(vert_res+1)*k+j));
			indices.append((vert_res+1+(vert_res+1)*k+j,0+(vert_res+1)*k+j,vert_res+2+(vert_res+1)*k+j));
	return (vertices,tcoords,indices)

class ParseError(Exception):
	"""AAM file has different format than expected"""
	def __init__(self, str,line):
		self.__problem = str
		self.__cur = line
	def __str__(self):
		return "%s: %s (line %d)" % (self.__doc__,self.__problem,self.__cur)

class NotSupported(Exception):
	"""Unsupported feature"""
	def __init__(self, str):
		self.__feat = str
	def __str__(self):
		return self.__feat + " " + self.__doc__

class AAMParser:
	"""Parser for AAM files"""
	def __init__(self,infile):
		self.infile = infile
		self.line = 0 # current line apart for BINARY file for which it is not really correct
		self.binary = False # BINARY files
		self.back = None # backward
	def readline(self):
		if not self.back is None:
			r = self.back
			self.back = None
			return r
		self.line = self.line + 1
		return self.infile.readline().replace("-- SIMO: aggiunti i 3 valori di Scale","")
	def unread(self,line):
		self.back = line
	def needed(self,what):
		l = self.readline()
		if l.strip() != what:
			raise ParseError("Waiting for %s <%s>"% (what,l),self.line)
	def readtoken(self,what=None):
		line = ""
		a = []
		while True:
			line = self.readline()
			if line == "":
				return None
			line = line.strip()
			if line == "":
				continue
			if line.find(":") >= 0:
				a = [x.strip() for x in line.split(":",1)]
			else:
				a = [x.strip() for x in line.split(" ",1)]
				if len(a) == 1:
					a.append("")
			if what is None or a[0] in what:
				return a
		if what:
			raise ParseError("Waiting for %s <%s><%s>" % (str(what),line,str(a)),self.line)
	def readfloats(self,count):
		return numpy.array(struct.unpack("%df" % count,self.infile.read(count*4)),dtype="float32")
	def readtokens(self,what):
		while True:
			tkn = self.readtoken()
			if not tkn:
				return None
			elif tkn[0] in what:
				return tkn
		raise ParseError("Waiting for " + str(what),self.line)
class BaseImage:
    def getSize(self,name): 
        pass
    def resize(self,name,size): 
        pass

class DDSImage(BaseImage):
    def getSize(self,name): 
        pass
    def resize(self,name,size): 
        pass

class PILImage(BaseImage):
    def getSize(self,name): 
        pass
    def resize(self,name,size): 
        pass

def getImageKit(name):
    if name.lower().endswith(".dds"):
        return DDSImage(name)
    else:
        return PILImage(name)

class Texture:
    def __init__(self):
        self.filename = ""
        self.filtering = "PYRAMIDAL"
        self.channel = 0
        self.intensity = 0
        self.rotationU = 0
        self.rotationV = 0
        self.rotationW = 0
        self.offsetU = 0
        self.offsetV = 0
        self.tilingU = 0
        self.tilingV = 0
    def read(self,parser):
        while True:
            what = parser.readtoken()
            if not what or what[0] == "}":
                break
            n = what[0]
            v = what[1]
            if n == "FN":
                self.filename = v
            elif n == "Fi":
                self.filtering = v
            elif n == "Ch":
                self.channel = float(v)
            elif n == "In":
                self.intensity = float(v)
            elif n == "UA":
                self.rotationU  = float(v)
            elif n == "VA":
                self.rotationV  = float(v)
            elif n == "WA":
                self.rotationW  = float(v)
            elif n == "UO":
                self.offsetU  = float(v)
            elif n == "VO":
                self.offsetV = float(v)
            elif n == "UT":
                self.tilingU  = float(v)
            elif n == "VT":
                self.tilingV = float(v)
            # unknown
    def write(self,out):
        out.write("\t{\r\n\t\tFN: %s\r\n\t\tFi: %s\r\n\t\tIn: %f\r\n\t\tCh: %d\r\n\t\tUA: %f\r\n\t\tVA: %f\r\n\t\tWA: %f\r\n\t\tUO: %f\r\n\t\tVO: %f\r\n\t\tUT: %f\r\n\t\tVT: %f\r\n}\t\r\n"
            % (self.filename,self.filtering,self.channel,self.intensity,self.rotationU,self.rotationV,self.rotationW,self.offsetU,self.offsetV,self.tilingU,self.tilingV))
class Material:
	def __init__(self):
		self.name = "01 - Default"
		self.matclass = "Standard"
		self.ambient = (1.0,1.0,1.0)
		self.diffuse = (1.0,1.0,1.0)
		self.specular = (1.0,1.0,1.0)
		self.transparency = 0.0
		self.shining = 0.0
		self.textures = []
		self.id = 0
		self.materials = []
		self.props ={}
		pass
	def write(self,outfile):
		outfile.write("\tName: %s\r\n\tClass: %s\r\n" % (self.name,self.matclass))
		outfile.write("\tAm: %s\r\n" % (" ".join([str(x) for x in self.ambient])))
		outfile.write("\tDi: %s\r\n" % (" ".join([str(x) for x in self.diffuse])))
		outfile.write("\tSp: %s\r\n" % (" ".join([str(x) for x in self.specular])))
		outfile.write("\tTr: %f\r\n" % self.transparency)
		outfile.write("\tSh: %f\r\n" % self.shining)
		outfile.write("\tWS: 1.0\r\n")
		outfile.write("\tSh: BLINN\r\n")
		# TODO textures
		if len(self.textures) > 0:
			outfile.write("\tTx: Y\r\n")
			self.textures[0].write(outfile)
		else:
			outfile.write("\tTx: N\r\n")
		if len(self.textures) > 1:
			outfile.write("\tTS: Y\r\n")
			self.textures[1].write(outfile)
		else:
			outfile.write("\tTS: N\r\n")
		
	def read(self,parser):
		while True:
			what = parser.readtoken()
			if what[0] == "}":
				return
			elif what[0] == "{":
				tex = Texture()
				tex.read(parser)
				self.textures.append(tex)
			elif what[0] == "Sub":
				parser.readtoken("{")
				matname = parser.readtoken("Name")
				matclass = parser.readtoken("Class")[1]
				if matclass == "Standard" or matclass == "Color":
					mat = Material()
				elif matclass == "Shell Material" or matclass == "Multi":
					mat = MultiMaterial()
				else:
					raise ParseError("Unknown Class %s " % matclass,parser.line)
				mat.matclass = matclass
				mat.name = matname[1]
				mat.read(parser)
				mat.id = len(self.materials)
				self.materials.append(mat)
			else:
				self.props[what[0]] = what[1:]

class MultiMaterial(Material):
	def __init__(self):
		Material.__init__(self)
		self.matclass = "Multi"

#TODO: normals using smoothing groups

class Triangle:
	def __init__(self,ii,ti,s):
		self.indices = ii
		self.texIndices = ti
		self.smoothing = s

class BoundingBox:
	def __init__(self,minx,maxx):
		if minx is None:
			self.min = None
			self.max = None
		else:
			self.min = minx
			self.max = maxx
	def __add__(self,other):
		if self.min is None:
			return other
		elif other.min is None:
			return self
		else:
			return BoundingBox(numpy.minimum(self.min,other.min),numpy.maximum(self.max,other.max))
class AAMGroup:
	def __init__(self):
		self.material = None
		self.triangles = []
		self.maxtexcoords = 0

		# restructured ONLY
		self.tris = None # all tri
		self.gvertices = None # all vertices
		self.texcoords = None # all texcoords
		self.normals = None # all normals
		self.colors = None
		self.bbox = None
	def gennormals(self):
		"""Computes the normals of the given group"""
		if self.gvertices is None:
			return
		self.normals = [(0,0,0) for i in range(0,len(self.gvertices))] # can be optimized?
		count = [0 for i in range(0,len(self.vertices))] # can be optimized?
		self.tnormals = [] 
		for i in range(0,len(self.tris)):
			v0,v1,v2 = [self.vertices[ivtx.v] for ivtx in self.tris[i]]
			d1 = vsub(v1,v0) 
			d2 = vsub(v2,v0) 
			n = vcross(d1,d2)
			n = vnormalize(n)
			self.tnormals.append(n)
		for i in range(0,len(self.tris)):
			n = self.tnormals[i]
			for ivtx in self.tris[i]:
				j = ivtx.v
				count[j] = count[j] + 1
				self.normals[j] = vadd(self.normals[j],n)
		for i in range(0,len(self.normals)):
			if count[i] > 0:
				self.normals[i] = vscale(self.normals[i],1.0/count[i])
	def restructure(self,base):
		vtx = []
		tex = []
		idx = []
		col = []
		skinw = []
		pairs = {}
		# TODO build multiple
		for i,tri in enumerate(self.triangles):
			qq = []
			for j in range(0,3): # every triangle vertex
				vt = tri[j].asTuple() # smooth + vertex + texture indices
				ivt = pairs.get(vt,None)
				if ivt is None:
					ivt = len(vtx)
					pairs[vt] = ivt
					vertexindex = vt[1]
					vtx.append(base.vertices[vertexindex,:])
					if len(base.colors) != 0:
						col.append(base.colors[vertexindex,:])

					effectivetexcoords = len(vt)-2
					self.maxtexcoords = effectivetexcoords
					
					# make tex contain not 2 values but 4
					if base.texcoords is not None and effectivetexcoords > 0:
						tei = []
						for j in range(0,effectivetexcoords):                    
							tei.append(base.texcoords[vt[j+2],:])
						tex.append(tei)
						#for j in range(n,self.maxtexcoords): # pad
						#	tei.extend([0,0])
					if base.skinweights is not None:
						while len(skinw) <= ivt:
							skinw.append([])
						skinw[ivt] = base.skinweights[vertexindex]  # TODO duplicate
				qq.append(TriangleVertex(ivt,[ivt]))
			idx.append(qq)
		minx = None
		maxx = None
		minx = vtx[0]
		maxx = vtx[0]
		for v in vtx:
			maxx = numpy.maximum(v,maxx)
			minx = numpy.minimum(v,minx)

		self.bbox = BoundingBox(minx,maxx)
		self.tris = idx
		self.vertices = vtx
		self.texcoords = tex
		self.colors = col
		self.skinweights = len(skinw) != 0 and skinw or None
		self.gennormals()
		print "restructured to",len(self.tris),"from",len(self.triangles)

class AAMFrame:
	def __init__(self):
		self.material = None
		self.vertices = None
		#self.texcoords = None
		self.aamObject = None
		self.animCtrl = None
		self.groups = None
def xfloat(x):
    if x == "-1.#IND":
        return 0.0
    else:
        return float(x)
class AAMPivot:
	def __init__(self,values):
		self.pos = [values[0],values[1],values[2]]
		self.axisrot = [values[3],values[4],values[5]]
		self.axisrad = values[6]

class AAMObject:
	def __init__(self):
		self.parent = -1
		self.parentobj = None
		self.children = []
		self.frames = []
		self.pivot = None # 3+4
		self.name = "Object"
		self.tm = None # 3x3 rotmat 3 pos 3 scal
		self.mid = -1  # Material ID
		self.material = None # material object
		self.mirror = False
		self.user = None # UserProp
		self.normals = []
		self.id = 0 # object id
		
		self.xstat = None
		self.ystat = None
		self.zstat = None
		self.bonespervtx = MAXWEIGHTS

		self.tricount = 0 # total triangle counts
		self.maxtexcoords = 0 # max tex coords in any group
		self.groups = [] # AAMGroup

		self.clist = 0
		self.colors = []
		self.vlist = 0 # vertex coordinates as (x,y,z)
		self.vertices = [] # global vertices
		self.tvlist = 0 # texture coordinates as (x,y)
		self.texcoords = [] # global texture coordinates
		self.skinweights = None
		self.boneid = None
		self.local2world = None
		self.tris = []
	def local2parent(self):
		if self.tm is None:
			return numpy.identity(4)
		else:
			# numpy is row based
			# TODO verify
			obj = self
			#onlyrot = numpy.array([obj.tm[0],obj.tm[3],obj.tm[6], obj.tm[1],obj.tm[4],obj.tm[7], obj.tm[2],obj.tm[5],obj.tm[8]]).reshape(3,3)
			noscale = numpy.array([obj.tm[0],obj.tm[3],obj.tm[6],obj.tm[9],    obj.tm[1],obj.tm[4],obj.tm[7],obj.tm[10],    obj.tm[2],obj.tm[5],obj.tm[8],obj.tm[11],   0,0,0,1]).reshape(4,4)
			if obj.tm[12] != 1 or obj.tm[13] != 1 or obj.tm[14] != 1:
				scale =  numpy.array([obj.tm[12],0,0,0,    0,obj.tm[13],0,0,   0,0,obj.tm[14],0,   0,0,0,1]).reshape(4,4)
				return noscale * scale
			else:
				return noscale

	def computelocal2world(self):		
		"""computes local 2 world matrix"""
		if self.parentobj is None or self.parentobj.local2world is None:
			base = numpy.identity(4)
		else:
			base = self.parentobj.local2world
		self.local2world = base * self.local2parent()
		for c in self.children:
			c.computelocal2world()		
	def reorder(self,backward = False):
		"""Applies recursively the vertex transformation (x,y,z)->(x,z,-y) forward or backward """
		if not backward:
			for i in range(0,len(self.vertices)):
				x,y,z = self.vertices[i]
				self.vertices[i] = [x,z,-y]
		else:
			for i in range(0,len(self.vertices)):
				x,y,z = self.vertices[i]
				self.vertices[i] = [x,-z,y]
	def gennormals(self):
		"""computes the normals of all the groups inside"""
		print self.id,self.name,"making normals of ",len(self.groups),"groups"
		for g in self.groups:
			g.gennormals()


	def write(self, outfile, iframe,iobj,redux):
		outfile.write("\tObj: %d %s\r\n\tPar: %d\r\n\t{\r\n" % (iobj,self.name,self.parent))
		if self.tm is not None:
			outfile.write("\tTM: %s\r\n" % (" ".join([str(x) for x in self.tm])))
		outfile.write("\tMatID: %d\r\n" % (self.mid))
		if not redux:
			outfile.write("\tV_List: %d\r\n" % (len(self.vertices)))
			for v in self.vertices:
				outfile.write("\t\t"+" ".join([str(x) for x in v])+"\r\n")
			for tv in self.texcoords:
				outfile.write("\tTV_List: %d\r\n" % (tv))
				for v in tv:
					outfile.write("\t\t"+" ".join([str(x) for x in v])+"\r\n")
			outfile.write("\tI_List: %d %d\r\n" % (len(self.tris),1))
			outfile.write("\t\tNEWGROUP: 0\r\n") # TODO
			# TODO update
			#if len(self.tristex) == len(self.tris):
			#    for i in range(0,len(self.tris)):
			#        outfile.write("\t\tI: " + " ".join([str(x) for x in self.tris[i]])+" 1\r\n")
			#        outfile.write("\t\tTI: " + " ".join([str(x) for x in self.tristex[i]])+"\r\n")
			#else:
			#    for t in self.tris:
			#        outfile.write("\t\tI: " + " ".join([str(x) for x in t])+" 1\r\n")
			outfile.write("\t\tENDGROUP\r\n")
		else:
			outfile.write("\tV_List: 0\r\n")
			outfile.write("\tTV_List: 0\r\n")
			outfile.write("\tI_List: 0 0\r\n")
			#outfile.write("\t\tNEWGROUP: 0\r\n")
			#outfile.write("\t\tENDGROUP\r\n")
			outfile.write("\tAnim_Ctrl: N\r\n")
		if self.skinweights is not None:
			outfile.write("\tSKIN: Y\r\n")
			for k,v in self.skinweights.iteritems():
				outfile.write("\t\t%d %s\r\n" % (k, " ".join(["%d %f" % (x[0],x[1]) for x in v])))
		outfile.write("\t}\r\n")
	def restructure(self):
		for g in self.groups:
			g.restructure(self)
		#print "output vertices:%d texturecoords:%d (%d) normals:%d faces:%d" % (len(self.vertices),len(self.texcoords),len(self.texcoords) > 0 and len(self.texcoords[0]) or 0,len(self.normals),len(self.tris))
	def writewebgl(self, output, iframe,iobj,redux):
		# TODO normals.binormals,...
		# TODO multiple texture coordinates with TI
		if not redux:
			output.append("{")
			output.append("id:%d," % self.id)
			if self.boneid is not None:
				output.append("boneid:%d," % self.boneid)
			output.append("parent:%d," % self.parent)
			output.append("name:\"%s\"," % self.name)
			if self.material is not None:
				m = self.material
				if len(m.textures) > 0 and m.textures[0].filename != "":
					print m.name,m.textures[0].filename
					output.append("texturename:\"%s\"," % m.textures[0].filename)
			output.append("matid:%d," % self.mid)
			if self.pivot is not None:
				output.append("pivot:[%s]," % ",".join([str(x) for x in self.pivot]))
			if self.tm is not None:
				output.append("transform:[%s]," % ",".join([str(x) for x in self.tm]))
			self.restructure()
			output.append("texcoords:%d," % self.maxtexcoords )
			print "TEx coords ",self.maxtexcoords
			if AAMShellCmd.interleavemode:
				olist = []
				if len(self.texcoords) == 0:
					output.append("vernor:[")
					for i in range(0,len(self.vertices)):
						olist.append(",".join([str(x) for x in self.vertices[i]]))
						olist.append(",".join([str(x) for x in self.normals[i]]))
				else:
					output.append("vernortex:[")
					for i in range(0,len(self.vertices)):
						olist.append(",".join([str(x) for x in self.vertices[i]]))
						olist.append(",".join([str(x) for x in self.normals[i]]))
						olist.append(",".join([str(x) for x in self.texcoords[i]]))
				output.append(",".join(olist))
				output.append("],")
				
				output.append("indices:[")
				olist = []
				for tri in self.tris:
					olist.append(",".join([str(ivtx) for ivtx in tri]))
				output.append(",".join(olist))
				output.append("],")
			else:
				output.append("vertices:[")
				olist = []
				for v in self.vertices:
					olist.append(",".join([str(x) for x in v]))
				output.append(",".join(olist))
				output.append("],")
				
				output.append("normals:[")
				olist = []
				for v in self.normals:
					olist.append(",".join([str(x) for x in v]))
				output.append(",".join(olist))
				output.append("],")
				
				output.append("texture:[")
				olist = []
				for v in self.texcoords:
					olist.append(",".join([str(x) for x in v]))
				output.append(",".join(olist))
				output.append("],")
				
				output.append("indices:[")
				olist = []
				for t in self.tris:
					olist.append(",".join([str(x) for x in t[0]]))
				output.append(",".join(olist))
				output.append("],")                
			if self.skinweights is not None:
				# compute bone indices
				# create two list: bone indices and bone weights
				tb = []
				tw = []
				for iv in range(0,len(self.vertices)):
					q = self.skinweights.get(iv,[])
					for b,w in q:
						tb.append(b)
						tw.append(w)
					for k in range(0,self.bonespervtx-len(q)):
						tb.append(0)
						tw.append(0)
				output.append("skin:{")
				output.append("bonecount:%d," % (max(tb)+1)) # TODO use bonemap    
				output.append("bones:[")    
				output.append(",".join([str(x) for x in tb]))
				output.append("],")    
				output.append("weights:[")    
				output.append(",".join([str(x) for x in tw]))
				output.append("],")    
				output.append("},")    
			output.append("}")
		else:
			output.append("{vertices:[],textures:[],indices:[]}")
	def writexml(self, output, iframe,iobj,redux):
		# TODO normals.binormals,...
		# TODO multiple texture coordinates with TI
		if not redux:
			self.restructure()
			output.write("<mesh id='%d' name='%s' materialid='%d'" % (self.id,self.name,self.mid))
			if self.skinweights is not None:
				tb = []
				tw = []
				for iv in range(0,len(self.vertices)):
					q = self.skinweights.get(iv,[])
					for b,w in q:
						tb.append(b)
						tw.append(w)
					for k in range(0,self.bonespervtx-len(q)):
						tb.append(0)
						tw.append(0)
				output.write(" isskin='1' bonecount='%d' objects='%d' bonespervertex='%d'" % (max(tb)+1,0,self.bonespervtx))
			if self.boneid is not None:
				output.write(" boneid='%d'" % self.boneid)
			if self.parent is not None:
				output.write(" parent='%d'" % self.parent)
			if self.parentboneid is not None:
				output.write(" parentboneid='%d'" % self.parentboneid)
			if self.xstat is not None:
				output.write(" bboxmin='%f %f %f'" % (self.xstat.min,self.ystat.min,self.zstat.min))
				output.write(" bboxmax='%f %f %f'" % (self.xstat.max,self.ystat.max,self.zstat.max))
			
			if self.material is not None:
				m = self.material
				if len(m.textures) > 0 and m.textures[0].filename != "":
					print m.name,m.textures[0].filename
					output.write(" texturename='%s' " % m.textures[0].filename)
			if self.pivot is not None:
				output.write(" pivot='%s'" % " ".join([str(x) for x in self.pivot]))
			if self.tm is not None:
				output.write(" transform='%s'" % " ".join([str(x) for x in self.tm]))
			output.write(" texcoords='%d'>" % self.maxtexcoords )
			print "TEx coords ",self.maxtexcoords
			if AAMShellCmd.interleavemode:
				z = []
				if len(self.texcoords) == 0:
					output.write("<vernor count='%d'>" % (len(self.vertices)*6))
					for i in range(0,len(self.vertices)):
						z.extend([str(x) for x in self.vertices[i]])
						z.extend([str(x) for x in self.normals[i]])
					output.write(" ".join(z))
					output.write("</vernor>")
				else:
					output.write("<vernortex count='%d'>" % (len(self.vertices)*8))
					for i in range(0,len(self.vertices)):
						z.extend([str(x) for x in self.vertices[i]])
						z.extend([str(x) for x in self.normals[i]])
						z.extend([str(x) for x in self.texcoords[i]])
					output.write(" ".join(z))
					output.write("</vernortex>\n")				
			else:
				output.write("<vertices count='%d'>" % (len(self.vertices)*3))
				for v in self.vertices:
						z.extend([str(x) for x in self.vertices[i]])
				output.write(" ".join(z))
				output.write("</vertices>\n")
				
				output.write("<normals count='%d'>" % (len(self.normals)*3))
				for v in self.normals:
					z.extend([str(x) for x in v])
				output.write(" ".join(z))
				output.write("</normals>\n")
				
				output.write("<texcoords count='%d'>" % (len(self.texcoords)*2))
				for v in self.texcoords:
					z.extend([str(x) for x in v])
				output.write(" ".join(z))
				output.write("</texcoords>\n")
				
			output.write("<indices>")
			for tri in self.tris:
				output.write(" ")
				output.write(" ".join([str(ivtx.v) for ivtx in tri]))
			output.write("</indices>")
			if self.skinweights is not None:
				# compute bone indices
				# create two list: bone indices and bone weights
				output.write("<bones>") # TODO use bonemap    
				output.write(" ".join([str(x) for x in tb]))
				output.write("</bones>\n")    
				output.write("<weights>")    
				output.write(" ".join([str(x) for x in tw]))
				output.write("</weights>\n")    
			output.write("</mesh>")
	def writexvr(self,iframe,iobj,redux):
		# TODO normals.binormals,...
		# TODO multiple texture coordinates with TI
		data = []
		if not redux:
			data.append(["id",self.id])
			if self.boneid is not None:
				data.append(["boneid",self.boneid])
			data.append(["name",self.name])
			data.append(["parent",self.parent])
			if self.material is not None:
				m = self.material
				if len(m.textures) > 0 and m.textures[0].filename != "":
					print m.name,m.textures[0].filename
					data.append(["texturename",m.textures[0].filename])
			data.append(["matid",self.mid])
			if self.pivot is not None:
				data.append(["pivot",asvector(self.pivot)])                
			if self.tm is not None:
				data.append(["transform",asvector(self.tm)])                
			self.restructure()
			data.append(["texcoords",self.maxtexcoords])                
			print "TEX coords ",self.maxtexcoords
			if AAMShellCmd.interleavemode:
				olist = []
				if len(self.normals) == 0:
					if len(self.groups) > 1:
						print "cannot use normals in more than 1 group",len(self.groups),"for",self.name
					else:
						if len(self.groups) > 0:
							self.normals = self.groups[0].normals
						if self.normals is None:
							self.normals = []
				if len(self.normals) == 0:
					ln = 0
				else:
					ln =      len(self.normals[0])
				if len(self.texcoords) == 0:
					vernor = []
					for i in range(0,len(self.vertices)):
						vernor.extend(self.vertices[i])
						if len(self.normals) == 0:
							vernor.extend((0,0,0,))
						else:
							vernor.extend(self.normals[i])
					data.append(["vernor",asvector(vernor)])        
					if len(self.vertices) > 0: 	
						l = (len(self.vertices[0]),ln)
					else:
						l = (0,ln)
					print l 
					data.append(["vernor_size",l])
				else:
					vernortex =[]
					for i in range(0,len(self.vertices)):
						vernortex.extend(self.vertices[i])
						if len(self.normals) == 0:
							vernortex.extend((0,0,0,))
						else:
							vernortex.extend(self.normals[i])
						vernortex.extend(self.texcoords[i])
					data.append(["vernortex",asvector(vernortex)])
					l = (len(self.vertices[0]),ln,len(self.texcoords[0]))
					print l
					data.append(["vernortex_size",l])
				indices = []
				olist = []
				for tri in self.tris:
					olist.extend([i.v for i in tri])
				data.append(["indices",intvectorasstr(olist)])                
			else:
				vertices = []
				for v in self.vertices:
					vertices.extend(v)
				data.append(["vertices",asvector(vertices)])
				
				normals = []
				for v in self.normals:
					normals.extend(v)
				data.append(["normals",asvector(normals)])                
				
				texcoords = []
				for v in self.texcoords:
					texcoords.extend(v)
				data.append(["texture",asvector(texcoords)])                
				
				indices = []
				for t in self.tris:
					indices.extend([i.v for i in t])
				data.append(["indices",intvectorasstr(indices)])                
			if self.skinweights is not None:
				# compute bone indices
				# create two list: bone indices and bone weights
				tb = []
				tw = []
				for iv in range(0,len(self.vertices)):
					q = self.skinweights.get(iv,[])
					for b,w in q:
						tb.append(b)
						tw.append(w)
					for k in range(0,self.bonespervtx-len(q)):
						tb.append(0)
						tw.append(0)
				data.append(["skin", [  ["bonepervertex",self.bonespervtx],["bonecount", max(tb)+1], ["bones", asvector(tb)], ["weights",asvector(tw)] ] ])
		else:
			print "redux not supported for writexvr"
		return data
	def read(self,parser):
		firstOut = True
		parser.readtoken("{")
		while True:
			what = parser.readtoken()
			if what is None:
				break
			if what[0] == "V_List":
				vlist = int(what[1])
				self.vlist = vlist
				if parser.binary:
					self.vertices = parser.readfloats(vlist*3).reshape((vlist,3),order="C")
				else:
					self.vertices = numpy.zeros((vlist,3),dtype="float32")
					for i in range(0,vlist):
						self.vertices[i,:] = [float(x) for x in parser.readline().strip().split(" ")]
			elif what[0] == "C_List":
				cvlist = int(what[1])
				self.cvlist = cvlist
				if parser.binary:
					self.colors = parser.readfloats(cvlist*3).reshape((vlist,3),order="C")
				else:
					self.colors = numpy.zeros((cvlist,3),dtype="float32")
					for i in range(0,cvlist):
						self.colors[i,:] = [float(x) for x in parser.readline().strip().split(" ")]
			elif what[0] == "SKIN":
				# vetrex_index numberofpairs bone_weight
				global MAXWEIGHTS
				if what[1] == "Y":
					self.skinweights = {}
					for i in range(0,self.vlist):
						line = parser.readline()
						pp = re.split(r'\s+', line.strip())
						values = [float(x) for x in pp if len(x) > 0]
						vtx = values[0]
						count = values[1]
						# transform list of bone,weight into list of (bone,weight)... and if bigger than required maximum reorder
						bws = [(values[j],values[j+1]) for j in range(2,len(values),2)] 
						self.skinweights[vtx] = sortandshorten(bws,MAXWEIGHTS)
			elif what[0] == "TV_List":
				tvlist = int(what[1])
				self.tvlist = tvlist
				if parser.binary:
					tv = parser.readfloats(tvlist*2).reshape((tvlist,2),order="C") # x y
				else:
					tv = numpy.zeros((tvlist,2),dtype="float32")
					for i in range(0,tvlist):
						tv[i,:] = [xfloat(x) for x in parser.readline().strip().split(" ")]
				self.texcoords = tv
				print self.name,"texcoordsobj",tv.shape[0]
			elif what[0] == "UserProp":
				self.user = what[1]
			elif what[0] == "LastPose":
				self.last = what[1]
			elif what[0] == "Anim_Ctrl":
				# TODO: if Y animation data follows
				if what[1] != "N":
					print "unsupported Anim_Ctrl"
					sys.exit(1)
			# ScSam RtSam PsSam RtIPA
			elif what[0] == "PsBez":
				continue
			elif what[0] == "ScBez":
				continue
			elif what[0] == "RtTcb":
				continue
			elif what[0] == "MatID":
				self.mid = int(what[1])
			elif what[0] == "TM":
				#the first nine values correspond to the rotation matrix, the next three values to the position, the last three values to the scaling along the three main axes
				self.tm = [float(x) for x in what[1].replace("\t"," ").split(" ") if x != ""]
			elif what[0] == "Piv":
				# 3+4
				self.pivot = [float(x) for x in what[1].replace("\t"," ").split(" ") if x != ""]
			elif what[0] == "Mir":
				self.mirror = int(what[1]) != 0
			elif what[0] == "I_List":
				if what[1].strip() == "0": # no data
					pass
				else:
					indicies,groups = [int(x) for x in what[1].strip().split(" ",1)]
					self.tricount = indicies
					mi = 0
					if parser.binary:
						for i in range(0,groups):
							q = parser.readtoken("NEWGROUP")
							g = AAMGroup()
							g.material = int(q[1])
							self.groups.append(g)
							lastI = None
							lastTI = []
							lastIS = 0
							groupmaxtexcoords = 0
							while True:
								x = parser.infile.read(1)
								if x == "I":
									if lastI is not None: # not the first
										groupmaxtexcoords = max(groupmaxtexcoords,len(lastTI))
										g.triangles.append(buildTriangle(lastI,lastTI,lastIS))
									w = parser.infile.read(2+4*4) # "I: " followed by 4 integers, but I is in x
									w = struct.unpack("4i",w[2:])
									lastIS = w[3]
									lastI = w[0:3]
									lastTI = []
								elif x == "T":
									w = parser.infile.read(3+3*4) # "TI: " followed by 3 integegers, but T is in x
									w = struct.unpack("3i",w[3:])
									lastTI.append(w) # TI: v1 v2 v3
								elif x == "E":
									parser.unread(x)
									break
								elif x == '\r' or x == '\n':
									continue
								else:
									if x != '\t':
										print "exit",ord(x)
									parser.unread(x+parser.readline())
									break
							q = parser.readtoken("ENDGROUP")
							if lastI is not None: # not the first
								groupmaxtexcoords = max(groupmaxtexcoords,len(lastTI))
								g.triangles.append(buildTriangle(lastI,lastTI,lastIS))
							print "donegroup tris:%d texs:%d" % (len(g.triangles),groupmaxtexcoords)
							g.maxtexcoords = groupmaxtexcoords
							mi = max(groupmaxtexcoords,mi)
					else:
						for i in range(0,groups):
							q = parser.readtoken("NEWGROUP")
							print "NEWGROUP",q
							g = AAMGroup()
							g.material = int(q[1])
							self.groups.append(g)
							# OPTION is groupid == submaterialid
							groupmaxtexcoords = 0
							lastI = None
							lastTI = []
							lastIS = 0
							while True:
								q = parser.readtoken()
								if q[0] == "I":
									if lastI is not None: # not the first
										groupmaxtexcoords = max(groupmaxtexcoords,len(lastTI))
										g.triangles.append(buildTriangle(lastI,lastTI,lastIS))
									w = [int(x) for x in q[1].strip().split(" ")] # TODO: smoothing group
									lastIS = w[3]
									lastI = w[0:3]
									lastTI = []
								elif q[0] == "TI":
									lastTI.append([int(x) for x in q[1].strip().split(" ")[0:3]])
								elif q[0] == "ENDGROUP":
									parser.unread("ENDGROUP")
									break
							q = parser.readtoken("ENDGROUP")
							if lastI is not None: # not the first
								groupmaxtexcoords = max(groupmaxtexcoords,len(lastTI))
								g.triangles.append(buildTriangle(lastI,lastTI,lastIS))
							mi = max(groupmaxtexcoords,mi)
					self.maxtexcoords = mi
					if mi != 0:							
						#print "maxtexcoords",mi
						pass
			elif what[0] == "}":
				break
			else:
				raise ParseError("unknown Object token %s" % (what[0]),parser.line)
	def __repr__(self):
		return "AAMobj: %f %f %f parent:%f material:%f" % (self.vlist,self.tricount,self.tvlist,self.parent,self.mid)
class AAM:
    def __init__(self,infile=None):
        self.materials = []
        self.objects = []
        self.frames = []
        self.shaders = []
        self.binary = False
        self.infile = None
        self.zip = None
        self.smoothed = False
        if infile:
            self.read(infile)
    def gennormals(self):
    	"""Computes the Normals of all the objects of the AAM file"""
    	for x in self.objects:
    		x.gennormals()
    def reorder(self,backward=False):
        for iobj in range(0,len(self.objects)):
            self.objects[iobj].reorder(backward)
    def findskin(self):
        for x in self.objects:
            if x.skinweights is not None:
                 return x
        return None
    def read(self,infile):
		fo = None
		if infile.endswith(".zip"):
			zip = zipfile.ZipFile(infile,"r")
			aams = [x for x in zip.namelist() if x.lower().endswith(".aam")]
			if len(aams) == 0:
				raise ParseError("no aam in zip file",0)
			elif len(aams) > 1:
				print "Warning: more than 1 file in AAM, getting first"
			fo = cStringIO.StringIO(zip.read(aams[0]))
			self.infile = aams[0]
			self.zip = zip
		else:
			self.infile = infile
			fo = open(infile,"rb")
		parser = AAMParser(fo)
		while True:
			what = parser.readtokens(set(["AAM_CHARACTER","AAM_MESH","SHADERS","BINARY","MATERIALS","GEOMETRY"]))
			if what is None:
				return
			elif what[0] == "AAM_CHARACTER":
				self.type = what[0]
				continue
			elif what[0] == "AAM_MESH":
				self.type = what[0]
				continue
			elif what[0] == "BINARY":
				parser.binary = True
				self.binary = True
				continue
			elif what[0] == "SHADERS":
				shaCount = int(parser.readtoken("num")[1])
				for i in range(0,shaCount):
					parser.readtoken("SHADERBEGIN")
					numPass = int(parser.readtoken("numPass")[1])
					while True:
						what = parser.readtoken()
						if not what or what[0] == "SHADEREND":
							break
				parser.readtoken("ENDSHADERS")
			elif what[0] == "MATERIALS":
				# do materials
				matCount = int(parser.readtoken("MatCount")[1])
				for i in range(0,matCount):
					what = parser.readtoken("Mat#")
					parser.readtoken("{")
					matname = parser.readtoken("Name")
					matclass = parser.readtoken("Class")[1]
					if matclass == "Standard" or matclass == "Color" or matclass == "Lambert" or matclass == "Phong":
						mat = Material()
					elif matclass == "Shell Material" or matclass == "Multi":
						mat = MultiMaterial()
					else:
						raise ParseError("unknown mat class %s " % matclass,parser.line)
					mat.matclass = matclass
					mat.name = matname[1]
					mat.read(parser)
					mat.id = len(self.materials)
					self.materials.append(mat)
				parser.needed("ENDMATERIALS")
				continue
			elif what[0] == "GEOMETRY":
				self.smoothed = len(what) > 0 and what[1] == "SmGEnabled"
				nobj = int(parser.readtoken("NObj")[1])
				nframes = int(parser.readtoken("NFrames")[1])
				parser.readtoken("Animation_mode")
				for i in range(0,nobj):
					self.objects.append(AAMObject())
				while True:
					what = parser.readtoken()
					if what[0] != "Frame":
						if what[0] != "ENDGEOMETRY":
							raise ParseError("Waiting for ENDGEOMETRY",parser.line)
						else:
							break
					f = AAMFrame()
					iframe = int(what[1])
					parser.needed("{")
					for i in range(0,nobj):
						q = parser.readtoken("Obj")[1].strip().split(" ",1)
						iobj,name = q
						obj = self.objects[int(iobj)]
						obj.name = name
						obj.parent = int(parser.readtoken("Par")[1])
						if obj.parent != -1:
							if len(self.objects) < obj.parent:
								print "missing ",obj.parent, "in",self.objects
							else:
								obj.parentobj = self.objects[obj.parent]
								obj.parentobj.children.append(obj)
						else:
							obj.parentobj = None
						obj.parentboneid = obj.parent
						obj.id = int(iobj)
						obj.boneid = obj.id
						obj.read(parser)
						#print "obj.id",obj.id," obj.mid ",obj.mid,self.materials[obj.mid].name
						obj.material = self.materials[obj.mid]
						obj.frames.append(f)
					parser.needed("}")
			else:
				raise ParseError("Waited GEOMETRY %s " % what,parser.line)
		print "ending..."
    def writewebgl(self,outfile,redux=False):
        # TODO materials
        # TODO multiple objects, for the moment only one
        # Output Structure
        # objects_sss =  array of dictionaries
        #     vertices:
        # 
        #
        # materials
        #SkelType:H-AnimLOA3
        output = ["output = ["]
        for iobj in range(0,len(self.objects)):
            self.objects[iobj].writewebgl(output,1,iobj,redux)
            output.append(",")
        output.append("]")
        outfile.write("\n".join(output))
        
    def writexml(self,outfile,redux=False):
		# TODO materials
		# TODO multiple objects, for the moment only one
		# Output Structure
		# objects_sss =  array of dictionaries
		#     vertices:
		# 
		#
		# materials
		#SkelType:H-AnimLOA3
		outfile.write("<meshes>");
		for iobj in range(0,len(self.objects)):
			self.objects[iobj].writexml(outfile,1,iobj,redux)
		outfile.write("</meshes>");
    def writexvr(self,redux=False):
        # TODO materials
        # TODO multiple objects, for the moment only one
        # Output Structure
        # objects_sss =  array of dictionaries
        #     vertices:
        # 
        #
        # materials
        #SkelType:H-AnimLOA3
        data = []
        for iobj in range(0,len(self.objects)):
            r = self.objects[iobj].writexvr(1,iobj,redux)
            data.append(r)
        return data
    def createbonemap(self,removeall=True):
		"""Maps the bone indices (as id of the objects) to a new list of bones for skipping the unused bones. First map the bones, then remap the skin"""
		oldboneid2boneid = {}

		# find the skin
		if removeall:
			for obj in self.objects:
				# skin
				if obj.skinweights is not None:				
					for iv in range(0,len(obj.vertices)):
						influencesiv = obj.skinweights.get(iv,[])
						for oldboneid,w in influencesiv:
							oldboneid2boneid.setdefault(oldboneid,len(oldboneid2boneid))
		else:
			for obj in self.objects:
				if obj.skinweights is None and obj.boneid is not None:
					oldboneid2boneid.setdefault(obj.boneid,len(oldboneid2boneid))
		print "Total Objects:",len(self.objects)
		print "Objects in vertexweights",oldboneid2boneid.keys()
		objid2boneid = {}
		#id2boneid[0] = 0
		for obj in self.objects:
			if obj.skinweights is None:
				# a possible bone
				k = oldboneid2boneid.get(obj.id,None)
				if k is not None:				
					obj.boneid = k
					objid2boneid[obj.id] = k
				else:
					obj.boneid = None				
			else:
				# fix skin
				obj.boneid = None
				for iv in range(0,len(obj.vertices)):
					obj.skinweights[iv] = [(oldboneid2boneid[oldbone],weight) for oldbone,weight in obj.skinweights.get(iv,[])]
		objid2boneid[-1] = -1
		for obj in self.objects:
			if len(objid2boneid) < obj.parent:
				obj.parentboneid = objid2boneid[obj.parent]
		if len(objid2boneid) > 0:
			print "bonemap is now long ",len(objid2boneid)-1,"vs original ",len(self.objects)
			print objid2boneid
		self.objid2boneid = oldboneid2boneid
    def write(self,outfile,redux=False):
        if self.binary:
            print "AAM binary output not supported"
        # materials
        #SkelType:H-AnimLOA3
        outfile.write("MATERIALS\r\n")
        outfile.write("MatCount: %d\r\n" % (len(self.materials)))
        for idx in range(0,len(self.materials)):
            outfile.write("Mat# %d\r\n{\r\n" % (idx))
            self.materials[idx].write(outfile)
            outfile.write("}\r\n")
        outfile.write("ENDMATERIALS\r\n")
        outfile.write("GEOMETRY SmGEnabled\r\n")
        outfile.write("NObj: %d\r\n" % (len(self.objects)))
        nframes = max([len(x.frames) for x in self.objects])
        outfile.write("NFrames: %d\r\n" % (nframes))
        if nframes > 1:
            outfile.write("Animation_mode: Keyframe\r\n")
        else:
            outfile.write("Animation_mode: None\r\n")
        for iframe in range(0,nframes):
            First = True
            for iobj in range(0,len(self.objects)):
                if iframe < len(self.objects[iobj].frames):
                    if First:
                        outfile.write("Frame: %d\r\n{\r\n" % iframe)
                        First = False
                    self.objects[iobj].write(outfile,iframe,iobj,redux)
            outfile.write("}\r\n")
        outfile.write("ENDGEOMETRY\r\n")
        
        #Animation_mode: Keyframe
        # geometry
        # SKIN: Y ... 0 1  1 1.0


        # - objects
        # - frames
        pass
def easysize(x):
    if x > 1024*1024:
        return "%dMB" % (x/(1024*1024))
    elif x > 1024:
        return "%dKB" % (x/1024)
    else:
        return "%dB" % x
import cmd
class AAMShellCmd(cmd.Cmd):
	saveindex = 1
	interleavemode = True
	def do_interleave(self,cmd):
		AAMShellCmd.interleavemode = True
	def do_nointerleave(self,cmd):
		AAMShellCmd.interleavemode = False
	def do_load(self,name):
		if name.endswith(".pickle"):
			self.aam = cPickle.load(open(name,"rb"))
			self.filename = name[0:-7]
			print self.filename
		else:
			self.aam = AAM(name)
			self.filename = name
		self.aam.size = 0
		try:
			self.aam.size = os.stat(name).st_size
		except:
			pass
	def do_info(self,text):
		a = self.aam
		print "AAM file: ",len(a.objects)," materials ",len(a.materials)," frames",a.frames," objects",a.objects
	def do_sizes(self,text):
		print "Mesh file size ", easysize(self.aam.size)
	def do_flipnormals(self,what):
		self.aam.gennormals()
		for o in self.aam.objects:
			if len(o.normals) > 0:
				n = o.normals;
				for i in range(0,len(n)):
					w = n[i]
					n[i] = [-w[0],-w[1],-w[2]]
			else:
				print "missing normals in do_flipnormals"
	def do_list(self,what):
		if what == "":
			print "list accepts: objs,mats,files"
		elif what == "objs":
			print "id name parentid matid vertices faces"
			for o in self.aam.objects:
				print o.id,o.name,o.parent,o.mid,o.vlist,o.tricount
		elif what == "mats":
			print "id name class"
			for m in self.aam.materials:
				print m.id,m.name,m.matclass,len(m.textures)
		elif what == "files":
			f = []
			for m in self.aam.materials:
				for km in m.materials:
					f = f + [t.filename for t in km.textures]
				f = f + [t.filename for t in m.textures]
			for s in self.aam.shaders:
				for p in s.passes:
					if p.name:
						f.append(p.name)
			print "\n".join(f)
	def do_bbox(self,part):
		for obj in self.aam.objects:
			xstat = Stats()        
			ystat = Stats()        
			zstat = Stats()        
			for p in obj.vertices:
				xstat.push(p[0])
				ystat.push(p[1])
				zstat.push(p[2])
			obj.xstat = xstat
			obj.ystat = ystat
			obj.zstat = zstat
			print obj.name,xstat,ystat,zstat
	def do_pickle(self,x):
		cPickle.dump(self.aam,open(self.filename+".pickle","wb"))
		print "pickled to ",self.filename+".pickle"
	def do_saveobj(self,part):
		if part == "":
			obj = self.aam.objects[0]
			print "Exporting First object ",obj.name
			outname = self.filename+ ".obj"
			self.saveobj(obj,outname)
		elif part == "all":
			self.saveobjrec(self.aam.objects)
		else:
			print [obj.name for obj in self.aam.objects]
			obj = [obj for obj in self.aam.objects if obj.name == part]
			if len(obj) == 0:
				print "Error: part not found"
				return
			obj = obj[0]
			outname = self.filename+ ".obj"
			self.saveobj(obj,outname)
		#http://local.wasp.uwa.edu.au/~pbourke/dataformats/obj/
	def saveobjrec(self,objs):
		for obj in objs:
			outname = self.filename + "-" + obj.name + ".obj"
			self.saveobj(obj,outname)
			self.saveobjrec(obj.children)		
	def saveobj(self,obj,outname):
		# NOTE: cannot store pivot or transforms in wavefront
		# NOTE: vertex color sometimes supported by adding them to vertex
		#
		obj.gennormals()
		obj.restructure()
		f = open(outname,"w")
		f.write("#object %d\n" % obj.id)
		if obj.parentobj is not None:
			pa = obj.parentobj.name
		else:
			pa = ""
		f.write("#parent %d %s\n" % (obj.parent,pa))
		f.write("#tm is %s\n" % str(obj.tm))
		f.write("#material id is %d\n" % obj.mid)
		f.write("#local2world is %s\n" % str(obj.local2world))
		f.write("mtllib %s\n" % (os.path.split(self.filename)[1] + ".mtl"))
		f.write("o %s\n" %obj.name)
		f.write("usemtl default\n")

		if len(obj.colors) != 0:
			print "Warning: vertex colors not supported by Wavefront Object"
		for v in obj.vertices:
			f.write("v %f %f %f\n" % (v[0],v[1],v[2]))
		for tv in obj.texcoords:
			f.write("vt %f %f\n" % (tv[0],tv[1]))

		for i,g in enumerate(obj.groups):
			if len(obj.groups) > 1:
				f.write("g group%d\n" % i)
				# TODO add "s ..."			
			for tv in g.tris:			
				if len(tv[0].t) > 0:
					if tv[0].n is not None:
						f.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (tv[0].v+1,tv[0].t[0]+1,tv[0].n+1,tv[1].v+1,tv[1].t[0]+1,tv[1].n+1,tv[2].v+1,tv[2].t[0]+1,tv[2].n+1))
					else:
						f.write("f %d/%d %d/%d %d/%d\n" % (tv[0].v+1,tv[0].t[0]+1,tv[1].v+1,tv[1].t[0]+1,tv[2].v+1,tv[2].t[0]+1))
				else:
					if tv[0].n is not None:
						f.write("f %d/%d %d/%d %d/%d\n" % (tv[0].v+1,tv[0].n+1,tv[1].v+1,tv[1].n+1,tv[2].v+1,tv[2].n+1))
					else:
						f.write("f %d %d %d\n" % (tv[0].v+1,tv[1].v+1,tv[2].v+1))

		#http://www.martinreddy.net/gfx/3d/OBJ.spec
		#http://local.wasp.uwa.edu.au/~pbourke/dataformats/mtl/

		#usemtl material_name
		#mtllib filename1 
	def do_gendom(self,outline):
		r = genskydom()
		a = AAM()
		o = AAMObject()
		o.vlist = len(r[0])
		o.tvlist = len(r[1])
		o.tricount = len(r[2])
		o.vertices = r[0]
		o.texcoords = r[1]
		o.tris = r[2]
		o.frames.append(0)
		o.mid = 0
		a.materials.append(Material())
		a.frames.append(0)
		a.objects.append(o)
		self.aam = a
	def do_save(self,outline):
		self.aam.write(open(outline == "" and "out%d.aam" % (self.saveindex) or outline,"wb"))
		self.saveindex = self.saveindex + 1
	def do_save0(self,outline):
		self.aam.write(open(outline == "" and "out%d.aam" % (self.saveindex) or outline,"wb"),True)
		self.saveindex = self.saveindex + 1
	def do_reorder(self,outline):
		self.aam.reorder(backward=False)
	def do_backreorder(self,outline):
		self.aam.reorder(backward=True)
	def do_savewebgl(self,outline):
		self.aam.gennormals()
		self.aam.createbonemap()
		self.aam.writewebgl(open(outline == "" and "out%d.js" % (self.saveindex) or outline,"wb"),False)
		self.saveindex = self.saveindex + 1
	def do_savexvr(self,outline):
		self.aam.gennormals()
		self.aam.createbonemap()        
		d = self.aam.writexvr()
		name = outline == "" and self.filename+".xvr" or outline
		f = open(name,"wb")
		netvars.encode(f,d)
		f.close()
		self.saveindex = self.saveindex + 1
		print "saved as ",name
	def do_savexml(self,outline):
		self.do_bbox("")
		self.aam.gennormals()
		#self.aam.createbonemap()
		self.aam.writexml(open(outline == "" and "out%d.xml" % (self.saveindex) or outline,"wb"),False)		
		self.saveindex = self.saveindex + 1
	def do_mergeobj(self,name):
		print "loading ",name
		o2 = SimpleOBJLoad(name)
		o2.restructure()
		skin = self.aam.findskin()
		print o2
		print "building KNN"
		kd = knn.buildKdHyperRectTree([PointVec3(skin.vertices[i],i) for i in range(0,len(skin.vertices))],10)
		sw = {}		
		firstout = True
		for i in range(0,len(o2.vertices)):
			neighbours = knn.Neighbors()
			neighbours.k = 4
			neighbours.points = []
			neighbours.minDistanceSquared = float("infinity")
			point = PointVec3(o2.vertices[i],i)
			knn.getKNN(point.data,kd,neighbours,knn.getFastDistance(kd.hyperRect.high,kd.hyperRect.low))
			pts = [neighbours.points[j][1] for j in range(0,neighbours.k)]
			# build now the new skinweights based on the point weights
			# for the moment just use one
			q = makeweights(pts,point)
			if False:
				sw[i] = skin.skinweights[q[0][0].baseIndex]
			else:
				allp = {}
				# all weights w 
				for p,w in q:
					for b,bw in skin.skinweights[p.baseIndex]:
						nw = allp.get(b,0) + bw*w
						allp[b] = nw
				if firstout:
					print "From bones ",len(allp.items())," to ",MAXWEIGHTS
					firstout = False
				sw[i] = shortenbws(allp.items(),MAXWEIGHTS) # (b,w)
				# first point, take point as 1.0
		skin.vertices = o2.vertices
		skin.normals = o2.normals
		skin.texcoords = o2.texcoords
		skin.skinweights = sw
		skin.tricount = len(o2.tris)
		skin.tris = o2.tris
		skin.gennormals()
		print "Merged as ",skin," ",len(skin.vertices),len(skin.skinweights),len(skin.texcoords),len(o2.tris)
		self.colors = []
	def do_leaftips(self,outline):
		# find leaves of device
		self.aam.createbonemap()
		asparent = set([x.parentboneid for x in self.aam.objects])
		bones = set([x.boneid for x in self.aam.objects])
		leaves = bones-asparent
		print "Bones",bones
		print "Parents",asparent
		print "Leaves",leaves
		leaveso = [x for x in self.aam.objects if x.boneid != 0 and x.boneid in leaves]
		skin = self.aam.objects[0]
		perbonevtx = defaultdict(set)
		for i in range(0,len(skin.vertices)):
			for b,w in skin.skinweights.get(i):
				perbonevtx[b].add(i)
		for x in leaveso:
			# all points with ...
			z = []
			pts = perbonevtx[x.boneid]
			print x.id,x.name,"pts",len(pts)
			if len(pts) == 0:
				pts = x.vertices
				print "use points from bone", len(pts)
				
			# now compute the sphere
	def do_stripskin(self,outline):
		# find skin
		# clean vertices
		# clean bones
		# clean rest
		for obj in self.aam.objects:
			if obj.skinweights is not None:
				obj.vlist = [(0,0,0)]
				obj.tvlist = [(0,0)]
				obj.tricount = 1
				obj.vertices = [(0,0,0)]
				obj.texcoords = [(0,0)]
				obj.colors = []
				obj.tris = [(0,0,0)]
				obj.normals = [] # per vertex
				obj.tnormals = [] # per triangle face
				obj.skinweights = { 0 : [(0,0)]}
	def do_tree(self,ops):
		# heavyweight but needed, then we could resolve in parser
		def recout(aam,t,s):
			ds = s + " "
			for x in t:
				print s,x.id, "/",x.parent," ",x.name," mat:",x.mid," ",len(x.vertices),"/",len(x.tris)
				recout(aam,[o for o in aam.objects if o.parent == x.id],ds)
		print "Tree"
		recout(self.aam,[o for o in self.aam.objects if o.parent == -1]," ")
	def do_help(self,t):
		print """
list (lists objects materials and files)
savewebgl outfile (output js with the content for webgl)
savecpp outfile (output cpp compatible)
quit (exits)
"""
	def do_quit(self,x):
		import sys
		sys.exit(-1)
def makeweights(ps,b):
	ds = [vlength(vsub(p.data,b.data)) for p in ps]
	for i in range(0,len(ds)):
		if ds[i] == 0:
			return [(ps[i],1.0)]
	dsi = [1.0/x for x in ds]
	t = sum(dsi)
	return [(ps[i],dsi[i]/t) for i in range(0,len(dsi))]
		
if __name__ == "__main__":
	import sys
	if len(sys.argv) == 1:
		print "AAM Shell by Emanuele Ruffaldi 2008-2009 %s " % __version__
	else:
		if sys.argv[1].endswith(".obj"):
			o = SimpleOBJLoad(sys.argv[1])
			print o
			o.restructure()
			print o
			if len(sys.argv) > 2:
				o2 = SimpleOBJLoad(sys.argv[2])
				print o2
				o2.restructure()
				print o2
				print "building knn from first"
				kd = knn.buildKdHyperRectTree([PointVec3(o.vertices[i],i) for i in range(0,len(o.vertices))],10)
				print "looking for second"
				for i in range(0,10): #len(o2.vertices)):
					neighbours = knn.Neighbors()
					neighbours.k = 4
					neighbours.points = []
					neighbours.minDistanceSquared = float("infinity")
					point = PointVec3(o2.vertices[i],i)
					knn.getKNN(point.data,kd,neighbours,knn.getFastDistance(kd.hyperRect.high,kd.hyperRect.low))
					pts = [neighbours.points[i][1] for i in range(0,neighbours.k)]
					print point,makeweights(pts,point)
		else:
			mycmd = AAMShellCmd()
			if sys.argv[1] != "new":
				mycmd.do_load(sys.argv[1])
			for i in sys.argv[2:]:
				mycmd.onecmd(i)
			mycmd.cmdloop()
