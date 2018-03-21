# -*- coding: utf-8 -*-


from __future__ import with_statement

import sys, os, datetime, urllib, optparse, contextlib

from ll import sisyphus, url, ul4c

from pycoco import xmlns


class File(object):
	def __init__(self, name):
		self.name = name
		self.lines = [] # list of lines with tuples (# of executions, line)

	def __repr__(self):
		return "<File name=%r at 0x%x>" % (self.name, id(self))


class Python_GenerateCodeCoverage(sisyphus.Job):
	def __init__(self, u, outputdir):
		sisyphus.Job.__init__(self, 2*60*60, name="Python_GenerateCodeCoverage", raiseerrors=1)
		self.url = url.URL(u)
		self.tarfile = self.url.file
		self.outputdir = url.Dir(outputdir)

		self.configurecmd = "./configure --enable-unicode=ucs4 --with-pydebug"

		self.gcovcmd = os.environ.get("COV", "gcov")
		self.makefile = "python/Makefile"

		self.buildlog = [] # the output of configuring and building Python
		self.testlog = [] # the output of running the test suite

	def cmd(self, cmd):
		self.logProgress(">>> %s" % cmd)
		pipe = os.popen(cmd + " 2>&1", "rb", 1)
		lines = []
		for line in pipe:
			self.logProgress("... " + line)
			lines.append(line)
		return lines

	def files(self, base):
		self.logProgress("### finding files")
		allfiles = []
		for (root, dirs, files) in os.walk(base):
			for file in files:
				if file.endswith(".py") or file.endswith(".c"):
					allfiles.append(File(os.path.join(root, file)))
		self.logProgress("### found %d files" % len(allfiles))
		return allfiles

	def download(self):
		self.logProgress("### downloading %s to %s" % (self.url, self.tarfile))
		urllib.urlretrieve(str(self.url), self.tarfile)

	def unpack(self):
		self.logProgress("### unpacking %s" % self.tarfile)
		self.cmd("tar xvjf %s" % self.tarfile)
		lines = list(open("python/.timestamp", "r"))
		self.timestamp = datetime.datetime.fromtimestamp(int(lines[0]))
		self.revision = lines[2]

	def configure(self):
		self.logProgress("### configuring")
		lines = self.cmd("cd python; %s" % self.configurecmd)
		self.buildlog.extend(lines)

	def make(self):
		self.logProgress("### running make")
		self.buildlog.extend(self.cmd("cd python && make coverage"))

	def test(self):
		self.logProgress("### running test")
		lines = self.cmd("cd python && ./python Lib/test/regrtest.py -T -N -uurlfetch,largefile,network,decimal")
		self.testlog.extend(lines)

	def cleanup(self):
		self.logProgress("### cleaning up files from previous run")
		self.cmd("rm -rf python")
		self.cmd("rm %s" % self.tarfile)

	def coveruncovered(self, file):
		self.logProgress("### faking coverage info for uncovered file %r" % file.name)
		file.lines = [(None, line.rstrip("\n")) for line in open(file.name, "r")]

	def coverpy(self, file):
		coverfilename = os.path.splitext(file.name)[0] + ".cover"
		self.logProgress("### fetching coverage info for Python file %r from %r" % (file.name, coverfilename))
		try:
			f = open(coverfilename, "r")
		except IOError, exc:
			self.logError(exc)
			self.coveruncovered(file)
		else:
			for line in f:
				line = line.rstrip("\n")
				prefix, line = line[:7], line[7:]
				prefix = prefix.strip()
				if prefix == "." or prefix == "":
					file.lines.append((-1, line))
				elif prefix == ">"*len(prefix):
					file.lines.append((0, line))
				else:
					file.lines.append((int(prefix.rstrip(":")), line))
			f.close()

	def coverc(self, file):
		self.logProgress("### fetching coverage info for C file %r" % file.name)
		dirname = os.path.dirname(file.name).split("/", 1)[-1]
		basename = os.path.basename(file.name)
		self.cmd("cd python && %s %s -o %s" % (self.gcovcmd, basename, dirname))
		try:
			f = open("python/%s.gcov" % basename, "r")
		except IOError, exc:
			self.logError(exc)
			self.coveruncovered(file)
		else:
			for line in f:
				line = line.rstrip("\n")
				if line.count(":") < 2:
					continue
				(count, lineno, line) = line.split(":", 2)
				count = count.strip()
				lineno = lineno.strip()
				if lineno == "0": # just the header
					continue
				if count == "-": # not executable
					file.lines.append((-1, line))
				elif count == "#####": # not executed
					file.lines.append((0, line))
				else:
					file.lines.append((int(count), line))
			f.close()

	def makehtml(self, files):
		# Generate main page
		self.logProgress("### generating index page")
		template = ul4c.compile(xmlns.page(xmlns.filelist(), onload="files_prepare()").conv().string())
		s = template.renders(
			filename=None,
			now=datetime.datetime.now(),
			timestamp=self.timestamp,
			revision=self.revision,
			crumbs=[
				dict(title="Core Development", href="http://www.python.org/dev/"),
				dict(title="Code coverage", href=None),
			],
			files=[
				dict(
					name=file.name.split("/", 1)[-1],
					lines=len(file.lines),
					coverablelines=sum(line[0]>=0 for line in file.lines),
					coveredlines=sum(line[0]>0 for line in file.lines),
				) for file in files
			],
		)
		u = self.outputdir/"index.html"
		with contextlib.closing(u.openwrite()) as f:
			f.write(s.encode("utf-8"))

		# Generate page for each source file
		template = ul4c.compile(xmlns.page(xmlns.filecontent()).conv().string())
		for (i, file) in enumerate(files):
			filename = file.name.split("/", 1)[-1]
			self.logProgress("### generating HTML %d/%d for %s" % (i+1, len(files), filename))
			s = template.renders(
				filename=filename,
				crumbs=[
					dict(title="Core Development", href="http://www.python.org/dev/"),
					dict(title="Code coverage", href="/index.html"),
					dict(title=filename, href=None),
				],
				lines=(
					dict(count=count, content=content.decode("latin-1").expandtabs(8)) for (count, content) in file.lines
				),
			)
			u = self.outputdir/(filename + ".html")
			with contextlib.closing(u.openwrite()) as f:
				f.write(s.encode("utf-8"))

		# Copy CSS/JS/GIF files
		for filename in ("coverage.css", "coverage_sortfilelist.css", "coverage.js", "spc.gif"):
			self.logProgress("### copying %s" % filename)
			try:
				import pkg_resources
			except ImportError:
				data = open(os.path.join(os.path.dirname(__file__), filename), "rb").read()
			else:
				data = pkg_resources.resource_string(__name__, filename)
			with contextlib.closing((self.outputdir/filename).openwrite()) as f:
				f.write(data)

		self.logProgress("### creating buildlog.txt")
		with contextlib.closing((self.outputdir/"buildlog.txt").openwrite()) as f:
			f.write("".join(self.buildlog))

		self.logProgress("### creating testlog.txt")
		with contextlib.closing((self.outputdir/"testlog.txt").openwrite()) as f:
			f.write("".join(self.testlog))

	def execute(self):
		self.cleanup()
		self.download()
		self.unpack()
		self.configure()
		files = self.files("python")
		self.make()
		self.test()
		for file in files:
			if file.name.endswith(".py"):
				self.coverpy(file)
			elif file.name.endswith(".c"):
				self.coverc(file)
		self.makehtml(files)
		self.logLoop("done with project Python (%s; %d files)" % (self.timestamp.strftime("%Y-%m-%d %H:%M:%S"), len(files)))


def main(args=None):
	p = optparse.OptionParser(usage="usage: %prog [options]")
	p.add_option("-u", "--url", dest="url", help="URL of the Python tarball", default="http://svn.python.org/snapshots/python3k.tar.bz2")
	p.add_option("-o", "--outputdir", dest="outputdir", help="Directory where to put the HTML files", default="~/pycoco")
	(options, args) = p.parse_args(args)
	if len(args) != 0:
		p.error("incorrect number of arguments")
		return 1

	sisyphus.execute(Python_GenerateCodeCoverage(options.url, options.outputdir))
	return 0


if __name__=="__main__":
	sys.exit(main())
