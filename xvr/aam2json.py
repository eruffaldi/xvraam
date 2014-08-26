#
# AAM to Pickled Geometry converter
# Emanuele Ruffaldi 2013
#
# This script takes a XVR-AAM model and makes it a pickled geometry for faster loading
import sys
import array
import json,numpy
from xvr.xvraam import *

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj,array.array):
        	return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class BinaryStore:
	def __init__(self,of):
		self.offset = 0
		self.of = of
	def packcoords(self,o):
		"""Generates numpy arrays of AAM vertex and face information"""
		if o.normals is not None:
			layout = ["pos","norm"]
		else:
			layout = ["pos"]

		if o.maxtexcoords == 0:
			pass
		else:
			layout.append(("texcoords", o.maxtexcoords,))
		stride = (3+3+o.maxtexcoords*2)
		if len(o.colors) != 0:
			stride += 3
			layout.append("color3")

		ss = (stride*len(o.vertices),1,)
		ad = numpy.zeros(ss,dtype="float32")
		k = 0
		# faces stored column major
		for i,v in enumerate(o.vertices):
			ad[k+0] = v[0]
			ad[k+1] = v[1]
			ad[k+2] = v[2]
			k += 3
			if o.normals is not None:
				n = o.normals[i]
				ad[k+0] = n[0]
				ad[k+1] = n[1]
				ad[k+2] = n[2]
				k += 3

			if len(o.texcoords) != 0:
				for t in o.texcoords[i]:
					ad[k+0] = t[0]
					ad[k+1] = t[1]
					k += 2

			if len(o.colors) != 0:
				c = o.colors[i]
				ad[k+0] = c[0]
				ad[k+1] = c[1]
				ad[k+2] = c[2]
				k += 3
		if self.of is not None:
			d = ad.tostring()
			block = dict(offset=self.offset,shape=ad.shape,type='f',size=len(d),items=len(d)/4,layout=layout,stride=stride)
			self.of.write(d)
			self.offset += len(d)
			ad = block

		# faces stored column major
		fd = numpy.zeros(3*len(o.tris),dtype="int32")
		k = 0
		for i,ft in enumerate(o.tris):
			fd[k+0] = ft[0].v
			fd[k+1] = ft[1].v
			fd[k+2] = ft[2].v
			k += 3
		if self.of is not None:
			d = fd.tostring()
			block = dict(offset=self.offset,shape=fd.shape,type='i',size=len(d),items=len(d)/4)
			self.of.write(d)
			self.offset += len(d)
			fd = block

		if o.skinweights is not None:
			maxsd = max([len(x) for x in o.skinweights])		
			sd = numpy.zeros(len(o.vertices)*maxsd*2,dtype="float32")
			k = 0
			for i in range(0,len(o.vertices)):
				ka = k
				for x in o.skinweights[i]:
					sd[ka] = x[0] 
					sd[ka+1] = x[1]
					ka += 2
				k += maxsd
			if self.of is not None:
				d = sd.tostring()
				block = dict(offset=self.offset,shape=sd.shape,type='i',size=len(d),items=len(d)/4)
				self.of.write(d)
				self.offset += len(d)
				sd = block
		else:
			sd = None
			maxsd = 0
		return (ad,fd,layout,stride*4,sd,maxsd)

def dumpmat(m):
	"""Exports an AAM material as dictionary"""
	md = {}
	md["subs"] = []
	md["id"] = m.id
	md["name"] = m.name
	md["class"] = m.matclass
	md["ambient"] = m.ambient
	md["diffuse"] = m.diffuse
	md["specular"] = m.specular
	md["shininess"] = m.shining
	md["transparency"] =  m.transparency
	md["textures"] = []
	for t in m.textures:
		md["textures"].append(dict(filename=t.filename,channel=t.channel))
	for sm in m.materials:
		print "\tsubmat",sm.id,sm.name
		md["subs"].append(dumpmat(sm))
	return md

if __name__ == "__main__":
	import sys
	import os
	import argparse
	import argparse

	parser = argparse.ArgumentParser(description='Converts AAM to json model')
	parser.add_argument('input', type=str, nargs='+',	                   help='multiple input files')
	parser.add_argument('--flat',action='store_true',help="flatten the geometry into a single mesh block")
	parser.add_argument('--verbose',action='store_true')
	parser.add_argument('--embed',action='store_true',default=False)
	args = parser.parse_args()

	for fm in args.input:
		if args.verbose:
			print "loading AAM:",fm
		aam = AAM(fm)
		dme = []  # collected objects
		dma = []  # collected materials
		if args.verbose:
			print "restructuring model and generating normals"
		for o in aam.objects:
			if args.verbose:
				print "processing %d %s mid:%d groups:%d" % (o.id,o.name,o.material.id,len(o.groups))
			o.restructure() # and gennormals
			o.computelocal2world()

		if args.flat:
			if args.verbose:
				print "aggregating geometry of the model"
			print "NOT IMPLEMENTED"

			# Idea:
			#   output "vertices and indices" of group become index in array
			#   precompute local2world transformation on all vertices and normals
			# Issues:
			#   one common layout, but some groups will not use them
			# 	KEY: no hierarchical processing but just one single pass over all groups, except optional hidden meshes (but not groups)
			#
			sys.exit(0)
		if args.embed:
			of = None
		else:
			of = open(os.path.splitext(fm)[0]+".bin","wb")
		bs = BinaryStore(of)

		for o in aam.objects:
			md = {}

			# transform tris
			# transform normals
			if args.verbose:
				print "generating %d %s" % (o.id,o.name)

			md["id"] = o.id
			md["name"] = o.name
			md["parent"] = o.parent
			md["materialid"] = o.material.id
			md["boneid"] = o.boneid
			md["parentboneid"] = o.parentboneid
			md["pivot"] = o.pivot
			md["local2parent"] = o.local2parent()
			md["local2world"] = o.local2world # useful for flat mode
			md["groups"] = []			
			bbox = BoundingBox(None,None) # empty
			if o.skinweights is not None:
				print "HAS!!!"
			for g in o.groups:
				ad,fd,layout,stride,sd,maxsd = bs.packcoords(g)
				bbox = g.bbox + bbox

				if args.verbose:
					print "\tgroup mat:%d vertices:%d tris:%d layout:%s" % (g.material,len(g.vertices),len(g.tris),str(layout))
				gd = {}
				gd["submaterialid"] = g.material # sub material
				gd["numvertices"] = len(g.vertices)	
				if maxsd > 0:	
					gd["numinfluences"] = maxsd
				gd["numtexcoords"] = g.maxtexcoords
				gd["bbox"] = dict(min=g.bbox.min,max=g.bbox.max)
		 		gd["stride"] = stride
				gd["layout"] = layout	
				gd["vertices"] = ad
				gd["tris"] = fd
				if maxsd > 0:	
					gd["skin"] = sd

				# TODO: numinfluences
				# TODO: bones

				md["groups"].append(gd)
			md["bbox"] = dict(min=bbox.min,max=bbox.max)
			dme.append(md)
		if args.verbose:
			print "procesing materials"
		for m in aam.materials:
			dma.append(dumpmat(m))
		if args.verbose:
			print "generating pickled file"
		dd = dict(objects=dme,materials=dma)
		open(os.path.splitext(fm)[0]+".json","wb").write(json.dumps(dd,cls=NumpyAwareJSONEncoder,sort_keys=True,                indent=4))


