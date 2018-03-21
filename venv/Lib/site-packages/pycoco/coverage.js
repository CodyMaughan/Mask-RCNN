function get_children(node, type)
{
	if (type)
		type = type.toLowerCase();
	var result = [];
	for (var i = 0; i < node.childNodes.length; ++i)
	{
		var child = node.childNodes[i];
		if (child.nodeType == 1 && (!type || (child.nodeName.toLowerCase() == type)))
			result.push(child);
	}
	return result;
}

function get_nth_child(node, type, count)
{
	if (!count)
		count = 0;
	if (type)
		type = type.toLowerCase();
	for (var i = 0; i < node.childNodes.length; ++i)
	{
		var child = node.childNodes[i];
		if (child.nodeType == 1 && (!type || (child.nodeName.toLowerCase() == type)))
			if (!count--)
				return child;
	}
	return null;
}

function get_class_child(node, type, class_)
{
	if (type)
		type = type.toLowerCase();
	for (var i = 0; i < node.childNodes.length; ++i)
	{
		var child = node.childNodes[i];
		if (child.nodeType == 1 &&
		    (!type || (child.nodeName.toLowerCase() == type)) &&
		    (!class_ || (child.getAttribute("class") == class_))
		)
			return child;
	}
	return null;
}

function get_first_text(node)
{
	while (node.nodeType != 3)
	{
		if (!node.childNodes.length)
			return null;
		node = node.childNodes[0];
	}
	return node.nodeValue;
}

function get_filename(tr)
{
	return get_first_text(get_nth_child(tr, "th", 0));
}

function get_nroflines(tr)
{
	return parseInt(get_first_text(get_nth_child(tr, "td", 0)), 10);
}

function get_coverablelines(tr)
{
	return parseInt(get_first_text(get_nth_child(tr, "td", 1)), 10);
}

function get_coveredlines(tr)
{
	return parseInt(get_first_text(get_nth_child(tr, "td", 2)), 10);
}

function get_coverage(tr)
{
	var content = get_first_text(get_nth_child(tr, "td", 3));
	if (content == "n/a")
		return -1.0;
	return parseFloat(content);
}

var files_rows = null;

function files_prepare()
{
	var tbody = document.getElementById("files");
	rows = get_children(tbody, "tr");
	for (var i = 0; i < rows.length; ++i)
	{
		var row = rows[i];
		row.filename = get_filename(row);
		row.nroflines = get_nroflines(row);
		row.coverablelines = get_coverablelines(row);
		row.coveredlines = get_coveredlines(row);
		row.coverage = get_coverage(row);
	}

	var button;
	
	button = document.createElement("span");
	button.innerHTML = "A-Z";
	if (document.all)
		button.onclick = function(){return files_sort(files_cmpbyfilename_asc, "filename");}
	else
		button.setAttribute("onclick", "return files_sort(files_cmpbyfilename_asc, 'filename');");
	document.getElementById("filename").appendChild(button);

	button = document.createElement("span");
	button.innerHTML = "Z-A";
	if (document.all)
		button.onclick = function(){return files_sort(files_cmpbyfilename_desc, "filename");}
	else
		button.setAttribute("onclick", "return files_sort(files_cmpbyfilename_desc, 'filename');");
	document.getElementById("filename").appendChild(button);
	
	button = document.createElement("span");
	button.innerHTML = "123";
	if (document.all)
		button.onclick = function(){return files_sort(files_cmpbynroflines_asc, "nroflines");}
	else
		button.setAttribute("onclick", "return files_sort(files_cmpbynroflines_asc, 'nroflines');");
	document.getElementById("nroflines").appendChild(button);

	button = document.createElement("span");
	button.innerHTML = "321";
	if (document.all)
		button.onclick = function(){return files_sort(files_cmpbynroflines_desc, "nroflines");}
	else
		button.setAttribute("onclick", "return files_sort(files_cmpbynroflines_desc, 'nroflines');");
	document.getElementById("nroflines").appendChild(button);
	
	button = document.createElement("span");
	button.innerHTML = "123";
	if (document.all)
		button.onclick = function(){return files_sort(files_cmpbycoverablelines_asc, "coverablelines");}
	else
		button.setAttribute("onclick", "return files_sort(files_cmpbycoverablelines_asc, 'coverablelines');");
	document.getElementById("coverablelines").appendChild(button);

	button = document.createElement("span");
	button.innerHTML = "321";
	if (document.all)
		button.onclick = function(){return files_sort(files_cmpbycoverablelines_desc, "coverablelines");}
	else
		button.setAttribute("onclick", "return files_sort(files_cmpbycoverablelines_desc, 'coverablelines');");
	document.getElementById("coverablelines").appendChild(button);
	
	button = document.createElement("span");
	button.innerHTML = "123";
	if (document.all)
		button.onclick = function(){return files_sort(files_cmpbycoveredlines_asc, "coveredlines");}
	else
		button.setAttribute("onclick", "return files_sort(files_cmpbycoveredlines_asc, 'coveredlines');");
	document.getElementById("coveredlines").appendChild(button);

	button = document.createElement("span");
	button.innerHTML = "321";
	if (document.all)
		button.onclick = function(){return files_sort(files_cmpbycoveredlines_desc, "coveredlines");}
	else
		button.setAttribute("onclick", "return files_sort(files_cmpbycoveredlines_desc, 'coveredlines');");
	document.getElementById("coveredlines").appendChild(button);

	button = document.createElement("span");
	button.innerHTML = "123";
	if (document.all)
		button.onclick = function(){return files_sort(files_cmpbycoverage_asc, "coverage");}
	else
		button.setAttribute("onclick", "return files_sort(files_cmpbycoverage_asc, 'coverage');");
	document.getElementById("coverage").appendChild(button);

	button = document.createElement("span");
	button.innerHTML = "321";
	if (document.all)
		button.onclick = function(){return files_sort(files_cmpbycoverage_desc, "coverage");}
	else
		button.setAttribute("onclick", "return files_sort(files_cmpbycoverage_desc, 'coverage');");
	document.getElementById("coverage").appendChild(button);
}

function files_sort(sorter, id)
{
	// Set wait cursor
	document.body.className = "wait";

	function dosort()
	{
		// Sort rows
		rows.sort(sorter);

		// Rearrange DOM according to sort result
		var tbody = document.getElementById("files");
		for (var i = 0; i < rows.length; ++i)
			tbody.appendChild(rows[i]);

		// Highlight sort column
		var ids = ["filename", "nroflines", "coverablelines", "coveredlines", "coverage"];
		var css = document.styleSheets[1];
		for (var i = 0; i < ids.length; ++i)
		{
			css.cssRules[i].style.backgroundColor = (ids[i] == id ? "#2f5a7e" : "#376a94");
			css.cssRules[i+5].style.backgroundColor = (ids[i] == id ? "#f2f2f2" : "#fff");
		}

		// Remove wait cursor
		document.body.className = "nowait";
	}

	// Start sort with timeout, so that new cursor can kick in
	window.setTimeout(dosort, 0.01);
	return false;
}

function files_cmpbyfilename_asc(tr1, tr2)
{
	var fn1 = tr1.filename;
	var fn2 = tr2.filename;
	return (fn1>fn2?1:0)-(fn1<fn2?1:0);
}

function files_cmpbyfilename_desc(tr1, tr2)
{
	var fn1 = tr1.filename;
	var fn2 = tr2.filename;
	return (fn1<fn2?1:0)-(fn1>fn2?1:0);
}

function files_cmpbynroflines_asc(tr1, tr2)
{
	return tr1.nroflines-tr2.nroflines;
}

function files_cmpbynroflines_desc(tr1, tr2)
{
	return tr2.nroflines-tr1.nroflines;
}

function files_cmpbycoverablelines_asc(tr1, tr2)
{
	return tr1.coverablelines-tr2.coverablelines;
}

function files_cmpbycoverablelines_desc(tr1, tr2)
{
	return tr2.coverablelines-tr1.coverablelines;
}

function files_cmpbycoveredlines_asc(tr1, tr2)
{
	return tr1.coveredlines-tr2.coveredlines;
}

function files_cmpbycoveredlines_desc(tr1, tr2)
{
	return tr2.coveredlines-tr1.coveredlines;
}

function files_cmpbycoverage_asc(tr1, tr2)
{
	return tr1.coverage-tr2.coverage;
}

function files_cmpbycoverage_desc(tr1, tr2)
{
	return tr2.coverage-tr1.coverage;
}
