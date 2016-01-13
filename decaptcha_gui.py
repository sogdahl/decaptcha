__author__ = 'Steven Ogdahl'

import importlib
import inspect
import os
from PIL import Image
import re

try:
    import wx
except ImportError:
    print "The wx module isn't available, so this module is unavailable for use."
    import sys
    sys.exit(1)

from decaptcha import pytesser, processing, flows

processors = {}
for processor in processing.processors:
    argspec = inspect.getargspec(processor)

    processors[processor.__name__] = {
        'name': processor.__name__,
        'method': processor
    }
    # Ignore the first parameter, which is always the image
    if argspec.args[1:]:
        processors[processor.__name__]['args'] = [{'name': a} for a in argspec.args[1:]]
        if argspec.defaults:
            default_offset = len(argspec.args[1:]) - len(argspec.defaults)
            for i in xrange(len(argspec.defaults)):
                default = argspec.defaults[i]
                if isinstance(default, basestring):
                    default = "'{0}'".format(default)
                processors[processor.__name__]['args'][default_offset + i]['default'] = default

defined_flows = {}
for m in os.listdir(os.path.join('decaptcha', 'flows')):
    if not m.endswith('.py'):
        continue
    try:
        module = importlib.import_module('.{0}'.format(m[:-3]), flows.__name__)
    except ImportError:
        continue
    key = module.__name__.split('.')[-1]
    if hasattr(module, 'steps'):
        defined_flows[key] = {'module': module, 'steps': module.steps, 'charset': getattr(module, 'charset', '')}


class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        self.dirname = ''
        self.processor_list = [p for p in processors.keys()]
        self.processor_list.sort()
        self.flow_list = [p for p in defined_flows.keys()]
        self.flow_list.sort()

        self.source_image = ''

        wx.Frame.__init__(self, parent, title=title)

        self.CreateStatusBar()

        filemenu = wx.Menu()
        menuOpen = filemenu.Append(wx.ID_OPEN, "&Open...", " Open an image to process")
        menuSaveSource = filemenu.Append(wx.ID_SAVEAS, "&Save Source...", " Save the source image")
        menuExit = filemenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu, "&File")
        self.SetMenuBar(menuBar)

        # Events.
        self.Bind(wx.EVT_MENU, self.OnOpen, menuOpen)
        self.Bind(wx.EVT_MENU, self.OnSave, menuSaveSource)
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)

        self.sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.imagePanel = wx.Panel(self)

        self.sourcePanel = wx.Panel(self.imagePanel)
        self.sourceImagePanel = wx.Panel(self.sourcePanel, style=wx.BORDER_SUNKEN)
        self.sourcePILImg = None
        image = wx.EmptyImage(274, 74, clear=False)
        self.sourceImage = wx.StaticBitmap(self.sourceImagePanel, wx.ID_ANY, wx.BitmapFromImage(image))

        self.sourceButtonPanel = wx.Panel(self.sourcePanel)
        self.execute = wx.Button(self.sourceButtonPanel, label='Execute')
        self.execute.SetToolTip(wx.ToolTip("Execute processor"))
        self.execute.Disable()
        self.Bind(wx.EVT_BUTTON, self.EvtExecute, self.execute)
        self.executeFlow = wx.Button(self.sourceButtonPanel, label='Exe Flow')
        self.executeFlow.SetToolTip(wx.ToolTip("Execute flow"))
        self.executeFlow.Disable()
        self.Bind(wx.EVT_BUTTON, self.EvtExecuteFlow, self.executeFlow)
        self.execute_and_promote = wx.Button(self.sourceButtonPanel, label='Ex && Pro')
        self.execute_and_promote.SetToolTip(wx.ToolTip("Execute processor & Promote output to source"))
        self.execute_and_promote.Disable()
        self.Bind(wx.EVT_BUTTON, self.EvtExecuteAndPromote, self.execute_and_promote)

        self.sourceButtonSizer = wx.BoxSizer(wx.VERTICAL)
        self.sourceButtonSizer.Add(self.execute, 0, wx.TOP)
        self.sourceButtonSizer.Add(self.executeFlow, 0, wx.TOP)
        self.sourceButtonSizer.Add(self.execute_and_promote, 0, wx.TOP)
        self.sourceButtonPanel.SetSizer(self.sourceButtonSizer)

        self.sourceSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sourceSizer.Add(self.sourceImagePanel, 0, wx.LEFT)
        self.sourceSizer.Add(self.sourceButtonPanel, 0, wx.LEFT)
        self.sourcePanel.SetSizer(self.sourceSizer)

        self.destPanel = wx.Panel(self.imagePanel)
        self.destImagePanel = wx.Panel(self.destPanel, style=wx.BORDER_SUNKEN)
        self.destPILImg = None
        image = wx.EmptyImage(274, 74, clear=False)
        self.destImage = wx.StaticBitmap(self.destImagePanel, wx.ID_ANY, wx.BitmapFromImage(image))

        self.destButtonPanel = wx.Panel(self.destPanel)
        self.promote = wx.Button(self.destButtonPanel, label='Promote')
        self.promote.SetToolTip(wx.ToolTip("Promote output to source"))
        self.promote.Disable()
        self.Bind(wx.EVT_BUTTON, self.EvtPromote, self.promote)
        self.run_ocr = wx.Button(self.destButtonPanel, label='Run OCR')
        self.run_ocr.SetToolTip(wx.ToolTip("Run OCR logic on output image"))
        self.run_ocr.Disable()
        self.Bind(wx.EVT_BUTTON, self.EvtRunOCR, self.run_ocr)

        self.destButtonSizer = wx.BoxSizer(wx.VERTICAL)
        self.destButtonSizer.Add(self.promote, 0, wx.TOP)
        self.destButtonSizer.Add(self.run_ocr, 0, wx.TOP)
        self.destButtonPanel.SetSizer(self.destButtonSizer)

        self.destSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.destSizer.Add(self.destImagePanel, 0, wx.LEFT)
        self.destSizer.Add(self.destButtonPanel, 0, wx.LEFT)
        self.destPanel.SetSizer(self.destSizer)

        self.imageSizer = wx.BoxSizer(wx.VERTICAL)
        self.imageSizer.Add(self.sourcePanel, 0, wx.TOP)
        self.imageSizer.Add(self.destPanel, 0, wx.TOP)
        self.imagePanel.SetSizer(self.imageSizer)

        self.processorsPanel = wx.Panel(self)
        self.processors = wx.ComboBox(self.processorsPanel, pos=(0, 0), size=(-1, -1), choices=self.processor_list, style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.Bind(wx.EVT_COMBOBOX, self.EvtParameters, self.processors)
        self.parameters = wx.TextCtrl(self.processorsPanel, pos=(0, 0), size=(-1, -1), style=wx.TE_MULTILINE)
        self.processorsSizer = wx.BoxSizer(wx.VERTICAL)
        self.processorsSizer.Add(self.processors, 0, wx.TOP)
        self.processorsSizer.Add(self.parameters, 1, wx.EXPAND)
        self.processorsPanel.SetSizer(self.processorsSizer)

        self.flowsPanel = wx.Panel(self)
        self.flows = wx.ComboBox(self.flowsPanel, pos=(0, 0), size=(-1, -1), choices=self.flow_list, style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.Bind(wx.EVT_COMBOBOX, self.EvtFlowSteps, self.flows)
        self.flowsteps = wx.TextCtrl(self.flowsPanel, pos=(0, 0), size=(-1, -1), style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.flowcharset = wx.TextCtrl(self.flowsPanel, pos=(0, 0), size=(-1, -1))
        self.flowsSizer = wx.BoxSizer(wx.VERTICAL)
        self.flowsSizer.Add(self.flows, 0, wx.TOP)
        self.flowsSizer.Add(self.flowsteps, 1, wx.EXPAND)
        self.flowsSizer.Add(self.flowcharset, 0, wx.EXPAND)
        self.flowsPanel.SetSizer(self.flowsSizer)

        self.sizer.Add(self.imagePanel, 0, wx.LEFT)
        self.sizer.Add(self.processorsPanel, 1, wx.EXPAND)
        self.sizer.Add(self.flowsPanel, 2.5, wx.EXPAND)
        self.SetSizerAndFit(self.sizer)

        self.Show()

    def OnOpen(self, event):
        """ Open a file"""
        dlg = wx.FileDialog(self, "Choose a file", self.dirname, "", "*.*", wx.OPEN)
        path = None
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
        dlg.Destroy()

        if path:
            self.source_image = path
            self.set_source_image(Image.open(path))

    def OnSave(self, event):
        if not self.source_image:
            return

        """ Save image"""
        dlg = wx.FileDialog(self, "Save file as", self.dirname, self.source_image, "*.*", wx.SAVE | wx.OVERWRITE_PROMPT)
        path = None
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
        dlg.Destroy()

        if path:
            self.source_image = path
            self.sourcePILImg.save(path)

    def OnExit(self, event):
        self.Close(True)

    def EvtExecute(self, event):
        method = self.processors.StringSelection
        kwargs = {}
        for kwarg in self.parameters.Value.split('\n'):
            mo = re.match(r'(?P<kw>[a-zA-Z_][_a-zA-Z0-9]*)=(?P<val>.+)', kwarg)
            if not mo:
                continue
            kwargs[mo.group('kw')] = eval(mo.group('val'))

        self.set_dest_image(processors[method]['method'](self.sourcePILImg, **kwargs))
        self.promote.Enable()
        self.run_ocr.Enable()

    def EvtExecuteFlow(self, event):
        m = self.flows.StringSelection
        im = self.sourcePILImg
        for step in defined_flows[m]['steps']:
            kwargs = {}
            if len(step) > 1:
                kwargs = step[1]
            self.set_source_image(im)
            im = getattr(processing, step[0])(im, **kwargs)
            self.set_dest_image(im)

        self.promote.Enable()
        self.run_ocr.Enable()

    def EvtParameters(self, event):
        if self.sourcePILImg:
            self.execute.Enable()
            self.execute_and_promote.Enable()
        self.parameters.Clear()
        if 'args' not in processors[event.GetString()]:
            return
        for p in processors[event.GetString()]['args']:
            self.parameters.AppendText('{0}={1}\n'.format(p['name'], p.get('default', '')))

    def EvtFlowSteps(self, event):
        self.executeFlow.Enable()
        self.flowsteps.Clear()
        for step in defined_flows[event.GetString()]['steps']:
            kwargs = {}
            if len(step) > 1:
                kwargs = step[1]
            self.flowsteps.AppendText('{0}({1})\n'.format(step[0], ', '.join(['{0}={1}'.format(*kv) for kv in kwargs.items()])))
        self.flowcharset.SetValue(defined_flows[event.GetString()]['charset'])

    def EvtPromote(self, event):
        self.set_source_image(self.destPILImg)

    def EvtExecuteAndPromote(self, event):
        self.EvtExecute(event)
        self.EvtPromote(event)

    def EvtRunOCR(self, event):
        text = re.sub(r'\s', '', pytesser.image_to_string(self.destPILImg, cleanup=True, charset=self.flowcharset.GetValue()))
        dlg = wx.MessageDialog(self, "Output from OCR: '{0}'".format(text), "OCR output", wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def set_source_image(self, imagePIL):
        self.sourcePILImg = imagePIL
        image = wx.EmptyImage(self.sourcePILImg.size[0], self.sourcePILImg.size[1])
        image.SetData(self.sourcePILImg.convert("RGB").tostring())
        image.Rescale(image.Width * 2, image.Height * 2)
        self.sourceImage.SetBitmap(wx.BitmapFromImage(image))
        self.sourceImage.SetClientSize(self.sourceImage.GetSize())

        if self.processors.StringSelection:
            self.execute.Enable()
            self.execute_and_promote.Enable()

        self.Refresh()
        self.SetSizerAndFit(self.sizer)

    def set_dest_image(self, imagePIL):
        self.destPILImg = imagePIL
        image = wx.EmptyImage(self.destPILImg.size[0], self.destPILImg.size[1])
        image.SetData(self.destPILImg.convert("RGB").tostring())
        image.Rescale(image.Width * 2, image.Height * 2)
        self.destImage.SetBitmap(wx.BitmapFromImage(image))
        self.destImage.SetClientSize(self.destImage.GetSize())

        self.Refresh()
        self.SetSizerAndFit(self.sizer)

if __name__ == '__main__':
    app = wx.App(False)
    frame = MainWindow(None, "DeCAPTCHA GUI test app")
    app.MainLoop()
