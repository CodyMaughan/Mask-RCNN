# -*- coding: utf-8 -*-

"""
This module is an XIST namespace used for formatting the HTML coverage report.
"""

import datetime

from ll.xist import xsc
from ll.xist.ns import xml, html, meta, htmlspecials, ul4


xmlns = "http://xmlns.python.org/coverage"


class page(xsc.Element):
	xmlns = xmlns

	class Attrs(xsc.Element.Attrs):
		class title(xsc.TextAttr): required = True
		class crumbs(xsc.TextAttr): required = True
		class onload(xsc.TextAttr): pass

	def convert(self, converter):
		e = xsc.Frag(
			xml.XML(), "\n",
			html.DocTypeXHTML10transitional(), "\n",
			html.html(
				html.head(
					meta.contenttype(),
					html.title(
						"Python code coverage",
						ul4.if_("filename"),
							": ", ul4.printx("filename"),
						ul4.else_(),
							" (", ul4.print_("timestamp.format('%Y-%m-%d')"), ")",
						ul4.end("if"),
					),
					meta.stylesheet(href="/coverage.css"),
					meta.stylesheet(href="/coverage_sortfilelist.css"),
					htmlspecials.javascript(src="/coverage.js"),
				),
				html.body(
					html.div(
						html.div(
							html.a(
								htmlspecials.autoimg(src="http://www.python.org/images/python-logo.gif", alt="Python", border=0),
								href="http://www.python.org/",
							),
							class_="logo",
						),
						html.div(
							ul4.for_("(i, item) in enumerate(crumbs)"),
								html.span(
									html.span(
										ul4.if_("i"),
											">",
										ul4.else_(),
											u"\xbb",
										ul4.end("if"),
										class_="bullet",
									),
									ul4.if_("item.href"),
										html.a(ul4.printx("item.title"), href=ul4.printx("item.href")),
									ul4.else_(),
										html.span(ul4.printx("item.title"), class_="here"),
									ul4.end("if"),
								ul4.end("for"),
								class_="crumb",
							),
							class_="crumbs",
						),
						class_="header",
					),
					html.div(
						self.content,
						class_="content",
					),
					onload=ul4.attr_if("get('onload')", ul4.printx("onload")),
				),
			),
		)
		return e.convert(converter)


class filelist(xsc.Element):
	xmlns = xmlns

	class Attrs(xsc.Element.Attrs):
		class timestamp(xsc.TextAttr): pass
		class revision(xsc.TextAttr): pass

	def convert(self, converter):
		now = datetime.datetime.now()
		e = xsc.Frag(
			html.h1("Python code coverage"),
			html.p("Generated at ", ul4.printx("now.format('%Y-%m-%d %H:%M:%S')"), class_="note"),
			html.p("Repository timestamp ", ul4.printx("timestamp.format('%Y-%m-%d %H:%M:%S')"), class_="note"),
			html.p(ul4.printx("revision"), class_="note"),
			html.p(html.a("Build log", href="buildlog.txt"), " ",html.a("Test log", href="testlog.txt"), class_="note"),
			htmlspecials.plaintable(
				html.thead(
					html.tr(
						html.th("Filename", id="filename"),
						html.th("# lines", id="nroflines"),
						html.th("# coverable lines", id="coverablelines"),
						html.th("# covered lines", id="coveredlines"),
						html.th("coverage", id="coverage"),
						html.th("distribution", id="distibution"),
					),
				),
				html.tbody(
					ul4.for_("file in files"),
						html.tr(
							html.th(
								html.a(
									ul4.printx("file.name"),
									href=(ul4.printx("file.name"), ".html"),
								),
								class_="filename",
							),
							html.td(
								ul4.printx("file.lines"),
								class_="nroflines",
							),
							html.td(
								ul4.printx("file.coverablelines"),
								class_="coverablelines",
							),
							html.td(
								ul4.printx("file.coveredlines"),
								class_="coveredlines",
							),
							html.td(
								ul4.if_("file.coverablelines"),
									ul4.printx("((100.*file.coveredlines)/file.coverablelines).format('.2f')"),
									"%",
								ul4.else_(),
									"n/a",
								ul4.end("if"),
								class_=(
									"coverage",
									ul4.if_("not file.coverablelines"),
										" disabled",
									ul4.end("if"),
								),
							),
							html.td(
								ul4.code("totalwidth = 100"),
								ul4.if_("file.coverablelines"),
									ul4.if_("file.coverablelines < file.lines"),
										ul4.code("width = int(1.*(file.lines-file.coverablelines)/file.lines*100)"),
										htmlspecials.pixel(src="/spc.gif", width=ul4.printx("width"), height=8, style="background-color: #ccc;"),
										ul4.code("totalwidth -= width"),
									ul4.end("if"),
									ul4.if_("file.coveredlines < file.coverablelines"),
										ul4.code("width = int(1.*(file.coverablelines-file.coveredlines)/file.lines*100)"),
										htmlspecials.pixel(src="/spc.gif", width=ul4.printx("width"), height=8, style="background-color: #f00;"),
										ul4.code("totalwidth -= width"),
									ul4.end("if"),
									ul4.if_("totalwidth"),
										htmlspecials.pixel(src="/spc.gif", width=ul4.printx("totalwidth"), height=8, style="background-color: #0c0;"),
									ul4.end("if"),
								ul4.else_(),
									htmlspecials.pixel(src="/spc.gif", width=ul4.printx("totalwidth"), height=8, style="background-color: #000;"),
								ul4.end("if"),
								class_="dist",
							),
							class_="files",
						),
					ul4.end("for"),
					id="files",
				),
				class_="files",
			)
		)
		return e.convert(converter)


class fileitem(xsc.Element):
	xmlns = xmlns

	class Attrs(xsc.Element.Attrs):
		class name(xsc.TextAttr): required = True
		class lines(xsc.IntAttr): required = True
		class coverablelines(xsc.IntAttr): required = True
		class coveredlines(xsc.IntAttr): required = True

	def convert(self, converter):
		lines = int(self.attrs.lines)
		coverablelines = int(self.attrs.coverablelines)
		coveredlines = int(self.attrs.coveredlines)

		distsize = (100, 8)
		if coverablelines:
			coverage = "%.02f%%" % (100.*coveredlines/coverablelines)
			coverageclass = "coverage"
			distribution = xsc.Frag()
			totalwidth = 0
			if coverablelines < lines:
				width = int(float(lines-coverablelines)/lines*distsize[0])
				distribution.append(htmlspecials.pixel(width=width, height=distsize[1], style="background-color: #ccc;"))
				totalwidth += width
			if coveredlines < coverablelines:
				width = int(float(coverablelines-coveredlines)/lines*distsize[0])
				distribution.append(htmlspecials.pixel(width=width, height=distsize[1], style="background-color: #f00;"))
				totalwidth += width
			if totalwidth < distsize[0]:
				width = distsize[0]-totalwidth
				distribution.append(htmlspecials.pixel(width=width, height=distsize[1], style="background-color: #0c0;"))
				totalwidth += width
		else:
			coverage = "n/a"
			coverageclass = "coverage disable"
			distribution = htmlspecials.pixel(width=distsize[0], height=distsize[1], style="background-color: #000;")

		e = html.tr(
			html.th(
				html.a(
					self.attrs.name,
					href=("root:", self.attrs.name, ".html"),
				),
				class_="filename",
			),
			html.td(
				lines,
				class_="nroflines",
			),
			html.td(
				coverablelines,
				class_="coverablelines",
			),
			html.td(
				coveredlines,
				class_="coveredlines",
			),
			html.td(
				coverage,
				class_=coverageclass,
			),
			html.td(
				distribution,
				class_="dist",
			),
			class_="files",
		)
		return e.convert(converter)


class filecontent(xsc.Element):
	xmlns = xmlns

	class Attrs(xsc.Element.Attrs):
		class name(xsc.TextAttr): required = True

	def convert(self, converter):
		e = xsc.Frag(
			html.h1("Python code coverage for ", ul4.printx("filename")),
			htmlspecials.plaintable(
				html.thead(
					html.tr(
						html.th("#"),
						html.th("count"),
						html.th("content"),
					),
				),
				html.tbody(
					ul4.for_("(i, line) in enumerate(lines)"),
						html.tr(
							html.th(ul4.print_("i+1")),
							html.td(
								ul4.if_("not isnone(line.count) and line.count >= 0"),
									ul4.printx("line.count"),
								ul4.else_(),
									"n/a",
								ul4.end("if"),
								class_="count",
							),
							html.td(ul4.printx("line.content"), class_="line"),
							class_=(
								ul4.attr_if("isnone(line.count) or line.count <= 0"),
								ul4.if_("isnone(line.count) or line.count < 0"),
									"uncoverable",
								ul4.elif_("not line.count"),
									"uncovered",
								ul4.end("if"),
							),
						),
					ul4.end("for"),
				),
				class_="file",
			)
		)
		return e.convert(converter)
