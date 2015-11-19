ddk = None

def lala():
	# global ddk
	print "ddk:", ddk

def alal():
	global ddk
	ddk = "Hello DDK"

def run():
	alal()
	lala()

run()