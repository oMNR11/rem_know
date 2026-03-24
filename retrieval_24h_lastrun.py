#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on March 24, 2026, at 16:27
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'retrieval_30min'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '',
    'session': 'retrieval_24h',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\Remember_Know\\retrieval_24h_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    if deviceManager.getDevice('prac_key') is None:
        # initialise prac_key
        prac_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_key',
        )
    if deviceManager.getDevice('recall_submit_2') is None:
        # initialise recall_submit_2
        recall_submit_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='recall_submit_2',
        )
    if deviceManager.getDevice('recog_key_2') is None:
        # initialise recog_key_2
        recog_key_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='recog_key_2',
        )
    if deviceManager.getDevice('source_key_2') is None:
        # initialise source_key_2
        source_key_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='source_key_2',
        )
    if deviceManager.getDevice('rk_response_2') is None:
        # initialise rk_response_2
        rk_response_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='rk_response_2',
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    if deviceManager.getDevice('recall_submit') is None:
        # initialise recall_submit
        recall_submit = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='recall_submit',
        )
    if deviceManager.getDevice('recog_key') is None:
        # initialise recog_key
        recog_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='recog_key',
        )
    if deviceManager.getDevice('source_key') is None:
        # initialise source_key
        source_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='source_key',
        )
    if deviceManager.getDevice('rk_response') is None:
        # initialise rk_response
        rk_response = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='rk_response',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions" ---
    text = visual.TextStim(win=win, name='text',
        text="In this phase, your memory for the definitions you saw and heard earlier will be tested. Some definitions will be OLD (from earlier) and some will be NEW.\nFor each definition, you will go through the following steps:\n1. Type the word that matches the definition and press ENTER.\n2. Indicate if the definition is OLD (Press Y) or NEW (Press N).\nIf you say the item is OLD, you will also be asked:\n3. Was it presented Visually (Press V) or Auditorily (Press A)?\n4. Do you 'Remember' specific details (Press 1), just 'Know' it's familiar (Press 2), or is it a 'Guess' (Press 3)?\n\nPress the SPACEBAR when you are ready to begin",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "instructions2" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='If you recognize an item, you will be asked if it is Type A or Type B.\n\nTYPE A (Press 1): You can think back and re-experience specific details (visual, auditory, or mental thoughts) from when you learned it.\nTYPE B (Press 2): The item is highly familiar, but you cannot re-experience anything specific from when you learned it.\n\nBoth types of memories are completely normal!\nPress "Space" to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    
    # --- Initialize components for Routine "prac_instructions" ---
    prac_inst = visual.TextStim(win=win, name='prac_inst',
        text='We will now do 6 practice trials. The examiner will sit with you to help clarify the difference between Type A and Type B if needed.\n\nPress the SPACEBAR to begin the practice.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_key = keyboard.Keyboard(deviceName='prac_key')
    
    # --- Initialize components for Routine "prac_cued_recall" ---
    recall_prompt_2 = visual.TextStim(win=win, name='recall_prompt_2',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    typed_response_2 = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         ori=0.0, pos=(0, -0.2), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='white',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='typed_response_2',
         depth=-2, autoLog=True,
    )
    recall_submit_2 = keyboard.Keyboard(deviceName='recall_submit_2')
    
    # --- Initialize components for Routine "prac_recall_fb" ---
    feedback_2 = visual.TextStim(win=win, name='feedback_2',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prac_recog" ---
    recog_prompt_2 = visual.TextStim(win=win, name='recog_prompt_2',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    recog_key_2 = keyboard.Keyboard(deviceName='recog_key_2')
    
    # --- Initialize components for Routine "prac_source" ---
    source_prompt_2 = visual.TextStim(win=win, name='source_prompt_2',
        text="Was this definition presented visually or auditorily?\n\nPress 'V' for VISUAL (Text on screen)\nPress 'A' for AUDITORY (Heard out loud)",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    source_key_2 = keyboard.Keyboard(deviceName='source_key_2')
    
    # --- Initialize components for Routine "prac_rem_know" ---
    rk_prompt_2 = visual.TextStim(win=win, name='rk_prompt_2',
        text="How do you remember this item?\n\nPress 1 for TYPE A (I remember specific details)\nPress 2 for TYPE B (I just know it's familiar)\nPress 3 for GUESS",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    rk_response_2 = keyboard.Keyboard(deviceName='rk_response_2')
    
    # --- Initialize components for Routine "main_exp_inst" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text='Practice complete! \nIf you have any questions about the difference between Type A and Type B, please ask the researcher now.\n\nPress SPACE to begin the real test',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # --- Initialize components for Routine "cued_recall" ---
    recall_prompt = visual.TextStim(win=win, name='recall_prompt',
        text=None,
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    typed_response = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         ori=0.0, pos=(0, -0.2), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='white',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='typed_response',
         depth=-2, autoLog=True,
    )
    recall_submit = keyboard.Keyboard(deviceName='recall_submit')
    
    # --- Initialize components for Routine "recall_feedback" ---
    feedback = visual.TextStim(win=win, name='feedback',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "recognition" ---
    recog_prompt = visual.TextStim(win=win, name='recog_prompt',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    recog_key = keyboard.Keyboard(deviceName='recog_key')
    
    # --- Initialize components for Routine "source_memory" ---
    source_prompt = visual.TextStim(win=win, name='source_prompt',
        text="Was this definition presented visually or auditorily?\n\nPress 'V' for VISUAL (Text on screen)\nPress 'A' for AUDITORY (Heard out loud)",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    source_key = keyboard.Keyboard(deviceName='source_key')
    
    # --- Initialize components for Routine "rem_know" ---
    rk_prompt = visual.TextStim(win=win, name='rk_prompt',
        text="How do you remember this item?\n\nPress 1 for TYPE A (I remember specific details)\nPress 2 for TYPE B (I just know it's familiar)\nPress 3 for GUESS",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    rk_response = keyboard.Keyboard(deviceName='rk_response')
    
    # --- Initialize components for Routine "end" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='You have completed this memory test.\n\nThank you for your time and focus!\n\nPress the SPACEBAR to exit and save your data.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[text, key_resp],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions2" ---
    # create an object to store info about Routine instructions2
    instructions2 = data.Routine(
        name='instructions2',
        components=[text_3, key_resp_4],
    )
    instructions2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # store start times for instructions2
    instructions2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions2.tStart = globalClock.getTime(format='float')
    instructions2.status = STARTED
    thisExp.addData('instructions2.started', instructions2.tStart)
    instructions2.maxDuration = None
    # keep track of which components have finished
    instructions2Components = instructions2.components
    for thisComponent in instructions2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions2" ---
    instructions2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions2,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions2" ---
    for thisComponent in instructions2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions2
    instructions2.tStop = globalClock.getTime(format='float')
    instructions2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions2.stopped', instructions2.tStop)
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "instructions2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "prac_instructions" ---
    # create an object to store info about Routine prac_instructions
    prac_instructions = data.Routine(
        name='prac_instructions',
        components=[prac_inst, prac_key],
    )
    prac_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for prac_key
    prac_key.keys = []
    prac_key.rt = []
    _prac_key_allKeys = []
    # store start times for prac_instructions
    prac_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prac_instructions.tStart = globalClock.getTime(format='float')
    prac_instructions.status = STARTED
    thisExp.addData('prac_instructions.started', prac_instructions.tStart)
    prac_instructions.maxDuration = None
    # keep track of which components have finished
    prac_instructionsComponents = prac_instructions.components
    for thisComponent in prac_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "prac_instructions" ---
    prac_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *prac_inst* updates
        
        # if prac_inst is starting this frame...
        if prac_inst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prac_inst.frameNStart = frameN  # exact frame index
            prac_inst.tStart = t  # local t and not account for scr refresh
            prac_inst.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_inst, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prac_inst.started')
            # update status
            prac_inst.status = STARTED
            prac_inst.setAutoDraw(True)
        
        # if prac_inst is active this frame...
        if prac_inst.status == STARTED:
            # update params
            pass
        
        # *prac_key* updates
        waitOnFlip = False
        
        # if prac_key is starting this frame...
        if prac_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prac_key.frameNStart = frameN  # exact frame index
            prac_key.tStart = t  # local t and not account for scr refresh
            prac_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prac_key.started')
            # update status
            prac_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(prac_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(prac_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if prac_key.status == STARTED and not waitOnFlip:
            theseKeys = prac_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _prac_key_allKeys.extend(theseKeys)
            if len(_prac_key_allKeys):
                prac_key.keys = _prac_key_allKeys[-1].name  # just the last key pressed
                prac_key.rt = _prac_key_allKeys[-1].rt
                prac_key.duration = _prac_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=prac_instructions,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            prac_instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prac_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prac_instructions" ---
    for thisComponent in prac_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prac_instructions
    prac_instructions.tStop = globalClock.getTime(format='float')
    prac_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prac_instructions.stopped', prac_instructions.tStop)
    # check responses
    if prac_key.keys in ['', [], None]:  # No response was made
        prac_key.keys = None
    thisExp.addData('prac_key.keys',prac_key.keys)
    if prac_key.keys != None:  # we had a response
        thisExp.addData('prac_key.rt', prac_key.rt)
        thisExp.addData('prac_key.duration', prac_key.duration)
    thisExp.nextEntry()
    # the Routine "prac_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    prac_trials = data.TrialHandler2(
        name='prac_trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('practice_retrieval.csv'), 
        seed=None, 
    )
    thisExp.addLoop(prac_trials)  # add the loop to the experiment
    thisPrac_trial = prac_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPrac_trial.rgb)
    if thisPrac_trial != None:
        for paramName in thisPrac_trial:
            globals()[paramName] = thisPrac_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPrac_trial in prac_trials:
        prac_trials.status = STARTED
        if hasattr(thisPrac_trial, 'status'):
            thisPrac_trial.status = STARTED
        currentLoop = prac_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_trial.rgb)
        if thisPrac_trial != None:
            for paramName in thisPrac_trial:
                globals()[paramName] = thisPrac_trial[paramName]
        
        # --- Prepare to start Routine "prac_cued_recall" ---
        # create an object to store info about Routine prac_cued_recall
        prac_cued_recall = data.Routine(
            name='prac_cued_recall',
            components=[recall_prompt_2, typed_response_2, recall_submit_2],
        )
        prac_cued_recall.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        recall_prompt_2.setText(definition + "\n\nType the word below and press ENTER to submit.")
        # Run 'Begin Routine' code from clear_box_2
        # Force the textbox to be completely empty at the start of every trial
        typed_response_2.text = ''
        # Set the prompt text dynamically
        recall_prompt_2.text = definition + "\n\nType the word below and press ENTER to submit.\n\n(If you cannot remember the word, leave the box blank and press ENTER to see the answer)."
        typed_response_2.reset()
        # create starting attributes for recall_submit_2
        recall_submit_2.keys = []
        recall_submit_2.rt = []
        _recall_submit_2_allKeys = []
        # store start times for prac_cued_recall
        prac_cued_recall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_cued_recall.tStart = globalClock.getTime(format='float')
        prac_cued_recall.status = STARTED
        thisExp.addData('prac_cued_recall.started', prac_cued_recall.tStart)
        prac_cued_recall.maxDuration = None
        # keep track of which components have finished
        prac_cued_recallComponents = prac_cued_recall.components
        for thisComponent in prac_cued_recall.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_cued_recall" ---
        prac_cued_recall.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPrac_trial, 'status') and thisPrac_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *recall_prompt_2* updates
            
            # if recall_prompt_2 is starting this frame...
            if recall_prompt_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                recall_prompt_2.frameNStart = frameN  # exact frame index
                recall_prompt_2.tStart = t  # local t and not account for scr refresh
                recall_prompt_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_prompt_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_prompt_2.started')
                # update status
                recall_prompt_2.status = STARTED
                recall_prompt_2.setAutoDraw(True)
            
            # if recall_prompt_2 is active this frame...
            if recall_prompt_2.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from clear_box_2
            if t < 0.5:
                typed_response_2.text = ''
            
            # *typed_response_2* updates
            
            # if typed_response_2 is starting this frame...
            if typed_response_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                typed_response_2.frameNStart = frameN  # exact frame index
                typed_response_2.tStart = t  # local t and not account for scr refresh
                typed_response_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(typed_response_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'typed_response_2.started')
                # update status
                typed_response_2.status = STARTED
                typed_response_2.setAutoDraw(True)
            
            # if typed_response_2 is active this frame...
            if typed_response_2.status == STARTED:
                # update params
                pass
            
            # *recall_submit_2* updates
            waitOnFlip = False
            
            # if recall_submit_2 is starting this frame...
            if recall_submit_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                recall_submit_2.frameNStart = frameN  # exact frame index
                recall_submit_2.tStart = t  # local t and not account for scr refresh
                recall_submit_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_submit_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_submit_2.started')
                # update status
                recall_submit_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recall_submit_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recall_submit_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if recall_submit_2.status == STARTED and not waitOnFlip:
                theseKeys = recall_submit_2.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                _recall_submit_2_allKeys.extend(theseKeys)
                if len(_recall_submit_2_allKeys):
                    recall_submit_2.keys = _recall_submit_2_allKeys[-1].name  # just the last key pressed
                    recall_submit_2.rt = _recall_submit_2_allKeys[-1].rt
                    recall_submit_2.duration = _recall_submit_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_cued_recall,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_cued_recall.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_cued_recall.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_cued_recall" ---
        for thisComponent in prac_cued_recall.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_cued_recall
        prac_cued_recall.tStop = globalClock.getTime(format='float')
        prac_cued_recall.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_cued_recall.stopped', prac_cued_recall.tStop)
        prac_trials.addData('typed_response_2.text',typed_response_2.text)
        # check responses
        if recall_submit_2.keys in ['', [], None]:  # No response was made
            recall_submit_2.keys = None
        prac_trials.addData('recall_submit_2.keys',recall_submit_2.keys)
        if recall_submit_2.keys != None:  # we had a response
            prac_trials.addData('recall_submit_2.rt', recall_submit_2.rt)
            prac_trials.addData('recall_submit_2.duration', recall_submit_2.duration)
        # the Routine "prac_cued_recall" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_recall_fb" ---
        # create an object to store info about Routine prac_recall_fb
        prac_recall_fb = data.Routine(
            name='prac_recall_fb',
            components=[feedback_2],
        )
        prac_recall_fb.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        feedback_2.setText("The correct word is:  " + word)
        # Run 'Begin Routine' code from feedback_logic_2
        # Clean up what the participant typed (remove spaces, make uppercase)
        # typed_response is the name of your Textbox component in the previous routine
        user_answer = typed_response_2.text.strip().upper()
        correct_answer = word.strip().upper()
        
        # If their typed answer perfectly matches the target word, SKIP this feedback screen
        if user_answer == correct_answer:
            continueRoutine = False
        # store start times for prac_recall_fb
        prac_recall_fb.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_recall_fb.tStart = globalClock.getTime(format='float')
        prac_recall_fb.status = STARTED
        thisExp.addData('prac_recall_fb.started', prac_recall_fb.tStart)
        prac_recall_fb.maxDuration = None
        # keep track of which components have finished
        prac_recall_fbComponents = prac_recall_fb.components
        for thisComponent in prac_recall_fb.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_recall_fb" ---
        prac_recall_fb.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisPrac_trial, 'status') and thisPrac_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_2* updates
            
            # if feedback_2 is starting this frame...
            if feedback_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_2.frameNStart = frameN  # exact frame index
                feedback_2.tStart = t  # local t and not account for scr refresh
                feedback_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_2.started')
                # update status
                feedback_2.status = STARTED
                feedback_2.setAutoDraw(True)
            
            # if feedback_2 is active this frame...
            if feedback_2.status == STARTED:
                # update params
                pass
            
            # if feedback_2 is stopping this frame...
            if feedback_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_2.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_2.tStop = t  # not accounting for scr refresh
                    feedback_2.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_2.stopped')
                    # update status
                    feedback_2.status = FINISHED
                    feedback_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_recall_fb,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_recall_fb.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_recall_fb.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_recall_fb" ---
        for thisComponent in prac_recall_fb.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_recall_fb
        prac_recall_fb.tStop = globalClock.getTime(format='float')
        prac_recall_fb.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_recall_fb.stopped', prac_recall_fb.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_recall_fb.maxDurationReached:
            routineTimer.addTime(-prac_recall_fb.maxDuration)
        elif prac_recall_fb.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "prac_recog" ---
        # create an object to store info about Routine prac_recog
        prac_recog = data.Routine(
            name='prac_recog',
            components=[recog_prompt_2, recog_key_2],
        )
        prac_recog.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        recog_prompt_2.setText(definition + "\n\nDid you see or hear this definition earlier?\n\nPress 'Y' for YES\nPress 'N' for NO")
        # create starting attributes for recog_key_2
        recog_key_2.keys = []
        recog_key_2.rt = []
        _recog_key_2_allKeys = []
        # Run 'Begin Routine' code from code_2
        recog_prompt_2.text = definition + "\n\nDid you see or hear this definition earlier?\n\nPress 'Y' for YES\nPress 'N' for NO"
        # store start times for prac_recog
        prac_recog.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_recog.tStart = globalClock.getTime(format='float')
        prac_recog.status = STARTED
        thisExp.addData('prac_recog.started', prac_recog.tStart)
        prac_recog.maxDuration = None
        # keep track of which components have finished
        prac_recogComponents = prac_recog.components
        for thisComponent in prac_recog.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_recog" ---
        prac_recog.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPrac_trial, 'status') and thisPrac_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *recog_prompt_2* updates
            
            # if recog_prompt_2 is starting this frame...
            if recog_prompt_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                recog_prompt_2.frameNStart = frameN  # exact frame index
                recog_prompt_2.tStart = t  # local t and not account for scr refresh
                recog_prompt_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recog_prompt_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recog_prompt_2.started')
                # update status
                recog_prompt_2.status = STARTED
                recog_prompt_2.setAutoDraw(True)
            
            # if recog_prompt_2 is active this frame...
            if recog_prompt_2.status == STARTED:
                # update params
                pass
            
            # *recog_key_2* updates
            waitOnFlip = False
            
            # if recog_key_2 is starting this frame...
            if recog_key_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                recog_key_2.frameNStart = frameN  # exact frame index
                recog_key_2.tStart = t  # local t and not account for scr refresh
                recog_key_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recog_key_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recog_key_2.started')
                # update status
                recog_key_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recog_key_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recog_key_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if recog_key_2.status == STARTED and not waitOnFlip:
                theseKeys = recog_key_2.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=False)
                _recog_key_2_allKeys.extend(theseKeys)
                if len(_recog_key_2_allKeys):
                    recog_key_2.keys = _recog_key_2_allKeys[-1].name  # just the last key pressed
                    recog_key_2.rt = _recog_key_2_allKeys[-1].rt
                    recog_key_2.duration = _recog_key_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_recog,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_recog.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_recog.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_recog" ---
        for thisComponent in prac_recog.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_recog
        prac_recog.tStop = globalClock.getTime(format='float')
        prac_recog.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_recog.stopped', prac_recog.tStop)
        # check responses
        if recog_key_2.keys in ['', [], None]:  # No response was made
            recog_key_2.keys = None
        prac_trials.addData('recog_key_2.keys',recog_key_2.keys)
        if recog_key_2.keys != None:  # we had a response
            prac_trials.addData('recog_key_2.rt', recog_key_2.rt)
            prac_trials.addData('recog_key_2.duration', recog_key_2.duration)
        # the Routine "prac_recog" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_source" ---
        # create an object to store info about Routine prac_source
        prac_source = data.Routine(
            name='prac_source',
            components=[source_prompt_2, source_key_2],
        )
        prac_source.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from source_branching_2
        # If they pressed 'n' (No) during Recognition, skip this routine entirely
        if recog_key_2.keys == 'n':
            continueRoutine = False
        # create starting attributes for source_key_2
        source_key_2.keys = []
        source_key_2.rt = []
        _source_key_2_allKeys = []
        # store start times for prac_source
        prac_source.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_source.tStart = globalClock.getTime(format='float')
        prac_source.status = STARTED
        thisExp.addData('prac_source.started', prac_source.tStart)
        prac_source.maxDuration = None
        # keep track of which components have finished
        prac_sourceComponents = prac_source.components
        for thisComponent in prac_source.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_source" ---
        prac_source.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPrac_trial, 'status') and thisPrac_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *source_prompt_2* updates
            
            # if source_prompt_2 is starting this frame...
            if source_prompt_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                source_prompt_2.frameNStart = frameN  # exact frame index
                source_prompt_2.tStart = t  # local t and not account for scr refresh
                source_prompt_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(source_prompt_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'source_prompt_2.started')
                # update status
                source_prompt_2.status = STARTED
                source_prompt_2.setAutoDraw(True)
            
            # if source_prompt_2 is active this frame...
            if source_prompt_2.status == STARTED:
                # update params
                pass
            
            # *source_key_2* updates
            waitOnFlip = False
            
            # if source_key_2 is starting this frame...
            if source_key_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                source_key_2.frameNStart = frameN  # exact frame index
                source_key_2.tStart = t  # local t and not account for scr refresh
                source_key_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(source_key_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'source_key_2.started')
                # update status
                source_key_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(source_key_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(source_key_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if source_key_2.status == STARTED and not waitOnFlip:
                theseKeys = source_key_2.getKeys(keyList=['v','a'], ignoreKeys=["escape"], waitRelease=False)
                _source_key_2_allKeys.extend(theseKeys)
                if len(_source_key_2_allKeys):
                    source_key_2.keys = _source_key_2_allKeys[-1].name  # just the last key pressed
                    source_key_2.rt = _source_key_2_allKeys[-1].rt
                    source_key_2.duration = _source_key_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_source,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_source.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_source.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_source" ---
        for thisComponent in prac_source.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_source
        prac_source.tStop = globalClock.getTime(format='float')
        prac_source.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_source.stopped', prac_source.tStop)
        # check responses
        if source_key_2.keys in ['', [], None]:  # No response was made
            source_key_2.keys = None
        prac_trials.addData('source_key_2.keys',source_key_2.keys)
        if source_key_2.keys != None:  # we had a response
            prac_trials.addData('source_key_2.rt', source_key_2.rt)
            prac_trials.addData('source_key_2.duration', source_key_2.duration)
        # the Routine "prac_source" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_rem_know" ---
        # create an object to store info about Routine prac_rem_know
        prac_rem_know = data.Routine(
            name='prac_rem_know',
            components=[rk_prompt_2, rk_response_2],
        )
        prac_rem_know.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from rk_branching_2
        # If they pressed 'n' (No) during Recognition, skip this routine entirely
        if recog_key_2.keys == 'n':
            continueRoutine = False
        # create starting attributes for rk_response_2
        rk_response_2.keys = []
        rk_response_2.rt = []
        _rk_response_2_allKeys = []
        # store start times for prac_rem_know
        prac_rem_know.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_rem_know.tStart = globalClock.getTime(format='float')
        prac_rem_know.status = STARTED
        thisExp.addData('prac_rem_know.started', prac_rem_know.tStart)
        prac_rem_know.maxDuration = None
        # keep track of which components have finished
        prac_rem_knowComponents = prac_rem_know.components
        for thisComponent in prac_rem_know.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_rem_know" ---
        prac_rem_know.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPrac_trial, 'status') and thisPrac_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rk_prompt_2* updates
            
            # if rk_prompt_2 is starting this frame...
            if rk_prompt_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                rk_prompt_2.frameNStart = frameN  # exact frame index
                rk_prompt_2.tStart = t  # local t and not account for scr refresh
                rk_prompt_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rk_prompt_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rk_prompt_2.started')
                # update status
                rk_prompt_2.status = STARTED
                rk_prompt_2.setAutoDraw(True)
            
            # if rk_prompt_2 is active this frame...
            if rk_prompt_2.status == STARTED:
                # update params
                pass
            
            # *rk_response_2* updates
            waitOnFlip = False
            
            # if rk_response_2 is starting this frame...
            if rk_response_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                rk_response_2.frameNStart = frameN  # exact frame index
                rk_response_2.tStart = t  # local t and not account for scr refresh
                rk_response_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rk_response_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rk_response_2.started')
                # update status
                rk_response_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(rk_response_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(rk_response_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if rk_response_2.status == STARTED and not waitOnFlip:
                theseKeys = rk_response_2.getKeys(keyList=['1','2','3'], ignoreKeys=["escape"], waitRelease=False)
                _rk_response_2_allKeys.extend(theseKeys)
                if len(_rk_response_2_allKeys):
                    rk_response_2.keys = _rk_response_2_allKeys[-1].name  # just the last key pressed
                    rk_response_2.rt = _rk_response_2_allKeys[-1].rt
                    rk_response_2.duration = _rk_response_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_rem_know,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_rem_know.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_rem_know.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_rem_know" ---
        for thisComponent in prac_rem_know.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_rem_know
        prac_rem_know.tStop = globalClock.getTime(format='float')
        prac_rem_know.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_rem_know.stopped', prac_rem_know.tStop)
        # check responses
        if rk_response_2.keys in ['', [], None]:  # No response was made
            rk_response_2.keys = None
        prac_trials.addData('rk_response_2.keys',rk_response_2.keys)
        if rk_response_2.keys != None:  # we had a response
            prac_trials.addData('rk_response_2.rt', rk_response_2.rt)
            prac_trials.addData('rk_response_2.duration', rk_response_2.duration)
        # the Routine "prac_rem_know" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisPrac_trial as finished
        if hasattr(thisPrac_trial, 'status'):
            thisPrac_trial.status = FINISHED
        # if awaiting a pause, pause now
        if prac_trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            prac_trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'prac_trials'
    prac_trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "main_exp_inst" ---
    # create an object to store info about Routine main_exp_inst
    main_exp_inst = data.Routine(
        name='main_exp_inst',
        components=[text_4, key_resp_3],
    )
    main_exp_inst.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for main_exp_inst
    main_exp_inst.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    main_exp_inst.tStart = globalClock.getTime(format='float')
    main_exp_inst.status = STARTED
    thisExp.addData('main_exp_inst.started', main_exp_inst.tStart)
    main_exp_inst.maxDuration = None
    # keep track of which components have finished
    main_exp_instComponents = main_exp_inst.components
    for thisComponent in main_exp_inst.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "main_exp_inst" ---
    main_exp_inst.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=main_exp_inst,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            main_exp_inst.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in main_exp_inst.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "main_exp_inst" ---
    for thisComponent in main_exp_inst.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for main_exp_inst
    main_exp_inst.tStop = globalClock.getTime(format='float')
    main_exp_inst.tStopRefresh = tThisFlipGlobal
    thisExp.addData('main_exp_inst.stopped', main_exp_inst.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "main_exp_inst" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_retrieval = data.TrialHandler2(
        name='trials_retrieval',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('retrieval_24hr.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trials_retrieval)  # add the loop to the experiment
    thisTrials_retrieval = trials_retrieval.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_retrieval.rgb)
    if thisTrials_retrieval != None:
        for paramName in thisTrials_retrieval:
            globals()[paramName] = thisTrials_retrieval[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_retrieval in trials_retrieval:
        trials_retrieval.status = STARTED
        if hasattr(thisTrials_retrieval, 'status'):
            thisTrials_retrieval.status = STARTED
        currentLoop = trials_retrieval
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_retrieval.rgb)
        if thisTrials_retrieval != None:
            for paramName in thisTrials_retrieval:
                globals()[paramName] = thisTrials_retrieval[paramName]
        
        # --- Prepare to start Routine "cued_recall" ---
        # create an object to store info about Routine cued_recall
        cued_recall = data.Routine(
            name='cued_recall',
            components=[recall_prompt, typed_response, recall_submit],
        )
        cued_recall.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        recall_prompt.setText('')
        # Run 'Begin Routine' code from clear_box
        # Force the textbox to be completely empty at the start of every trial
        typed_response.text = ''
        # Set the prompt text dynamically
        recall_prompt.text = definition + "\n\nType the word below and press ENTER to submit.\n\n(If you cannot remember the word, leave the box blank and press ENTER to see the answer)."
        typed_response.reset()
        # create starting attributes for recall_submit
        recall_submit.keys = []
        recall_submit.rt = []
        _recall_submit_allKeys = []
        # store start times for cued_recall
        cued_recall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        cued_recall.tStart = globalClock.getTime(format='float')
        cued_recall.status = STARTED
        thisExp.addData('cued_recall.started', cued_recall.tStart)
        cued_recall.maxDuration = None
        # keep track of which components have finished
        cued_recallComponents = cued_recall.components
        for thisComponent in cued_recall.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "cued_recall" ---
        cued_recall.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_retrieval, 'status') and thisTrials_retrieval.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *recall_prompt* updates
            
            # if recall_prompt is starting this frame...
            if recall_prompt.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                recall_prompt.frameNStart = frameN  # exact frame index
                recall_prompt.tStart = t  # local t and not account for scr refresh
                recall_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_prompt.started')
                # update status
                recall_prompt.status = STARTED
                recall_prompt.setAutoDraw(True)
            
            # if recall_prompt is active this frame...
            if recall_prompt.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from clear_box
            if t < 0.5:
                typed_response.text = ''
            
            # *typed_response* updates
            
            # if typed_response is starting this frame...
            if typed_response.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                typed_response.frameNStart = frameN  # exact frame index
                typed_response.tStart = t  # local t and not account for scr refresh
                typed_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(typed_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'typed_response.started')
                # update status
                typed_response.status = STARTED
                typed_response.setAutoDraw(True)
            
            # if typed_response is active this frame...
            if typed_response.status == STARTED:
                # update params
                pass
            
            # *recall_submit* updates
            waitOnFlip = False
            
            # if recall_submit is starting this frame...
            if recall_submit.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                recall_submit.frameNStart = frameN  # exact frame index
                recall_submit.tStart = t  # local t and not account for scr refresh
                recall_submit.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_submit, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_submit.started')
                # update status
                recall_submit.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recall_submit.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recall_submit.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if recall_submit.status == STARTED and not waitOnFlip:
                theseKeys = recall_submit.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                _recall_submit_allKeys.extend(theseKeys)
                if len(_recall_submit_allKeys):
                    recall_submit.keys = _recall_submit_allKeys[-1].name  # just the last key pressed
                    recall_submit.rt = _recall_submit_allKeys[-1].rt
                    recall_submit.duration = _recall_submit_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=cued_recall,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                cued_recall.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in cued_recall.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "cued_recall" ---
        for thisComponent in cued_recall.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for cued_recall
        cued_recall.tStop = globalClock.getTime(format='float')
        cued_recall.tStopRefresh = tThisFlipGlobal
        thisExp.addData('cued_recall.stopped', cued_recall.tStop)
        trials_retrieval.addData('typed_response.text',typed_response.text)
        # check responses
        if recall_submit.keys in ['', [], None]:  # No response was made
            recall_submit.keys = None
        trials_retrieval.addData('recall_submit.keys',recall_submit.keys)
        if recall_submit.keys != None:  # we had a response
            trials_retrieval.addData('recall_submit.rt', recall_submit.rt)
            trials_retrieval.addData('recall_submit.duration', recall_submit.duration)
        # the Routine "cued_recall" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "recall_feedback" ---
        # create an object to store info about Routine recall_feedback
        recall_feedback = data.Routine(
            name='recall_feedback',
            components=[feedback],
        )
        recall_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        feedback.setText("The correct word is:  " + word)
        # Run 'Begin Routine' code from feedback_logic
        # Clean up what the participant typed (remove spaces, make uppercase)
        # typed_response is the name of your Textbox component in the previous routine
        user_answer = typed_response.text.strip().upper()
        correct_answer = word.strip().upper()
        
        # If their typed answer perfectly matches the target word, SKIP this feedback screen
        if user_answer == correct_answer:
            continueRoutine = False
        # store start times for recall_feedback
        recall_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_feedback.tStart = globalClock.getTime(format='float')
        recall_feedback.status = STARTED
        thisExp.addData('recall_feedback.started', recall_feedback.tStart)
        recall_feedback.maxDuration = None
        # keep track of which components have finished
        recall_feedbackComponents = recall_feedback.components
        for thisComponent in recall_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_feedback" ---
        recall_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_retrieval, 'status') and thisTrials_retrieval.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback* updates
            
            # if feedback is starting this frame...
            if feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback.frameNStart = frameN  # exact frame index
                feedback.tStart = t  # local t and not account for scr refresh
                feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback.started')
                # update status
                feedback.status = STARTED
                feedback.setAutoDraw(True)
            
            # if feedback is active this frame...
            if feedback.status == STARTED:
                # update params
                pass
            
            # if feedback is stopping this frame...
            if feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback.tStop = t  # not accounting for scr refresh
                    feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback.stopped')
                    # update status
                    feedback.status = FINISHED
                    feedback.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=recall_feedback,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_feedback" ---
        for thisComponent in recall_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_feedback
        recall_feedback.tStop = globalClock.getTime(format='float')
        recall_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recall_feedback.stopped', recall_feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if recall_feedback.maxDurationReached:
            routineTimer.addTime(-recall_feedback.maxDuration)
        elif recall_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "recognition" ---
        # create an object to store info about Routine recognition
        recognition = data.Routine(
            name='recognition',
            components=[recog_prompt, recog_key],
        )
        recognition.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        recog_prompt.setText(definition + "\n\nDid you see or hear this definition earlier?\n\nPress 'Y' for YES\nPress 'N' for NO")
        # create starting attributes for recog_key
        recog_key.keys = []
        recog_key.rt = []
        _recog_key_allKeys = []
        # Run 'Begin Routine' code from code
        recog_prompt.text = definition + "\n\nDid you see or hear this definition earlier?\n\nPress 'Y' for YES\nPress 'N' for NO"
        # store start times for recognition
        recognition.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recognition.tStart = globalClock.getTime(format='float')
        recognition.status = STARTED
        thisExp.addData('recognition.started', recognition.tStart)
        recognition.maxDuration = None
        # keep track of which components have finished
        recognitionComponents = recognition.components
        for thisComponent in recognition.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recognition" ---
        recognition.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_retrieval, 'status') and thisTrials_retrieval.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *recog_prompt* updates
            
            # if recog_prompt is starting this frame...
            if recog_prompt.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                recog_prompt.frameNStart = frameN  # exact frame index
                recog_prompt.tStart = t  # local t and not account for scr refresh
                recog_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recog_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recog_prompt.started')
                # update status
                recog_prompt.status = STARTED
                recog_prompt.setAutoDraw(True)
            
            # if recog_prompt is active this frame...
            if recog_prompt.status == STARTED:
                # update params
                pass
            
            # *recog_key* updates
            waitOnFlip = False
            
            # if recog_key is starting this frame...
            if recog_key.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                recog_key.frameNStart = frameN  # exact frame index
                recog_key.tStart = t  # local t and not account for scr refresh
                recog_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recog_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recog_key.started')
                # update status
                recog_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recog_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recog_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if recog_key.status == STARTED and not waitOnFlip:
                theseKeys = recog_key.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=False)
                _recog_key_allKeys.extend(theseKeys)
                if len(_recog_key_allKeys):
                    recog_key.keys = _recog_key_allKeys[-1].name  # just the last key pressed
                    recog_key.rt = _recog_key_allKeys[-1].rt
                    recog_key.duration = _recog_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=recognition,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recognition.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recognition.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recognition" ---
        for thisComponent in recognition.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recognition
        recognition.tStop = globalClock.getTime(format='float')
        recognition.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recognition.stopped', recognition.tStop)
        # check responses
        if recog_key.keys in ['', [], None]:  # No response was made
            recog_key.keys = None
        trials_retrieval.addData('recog_key.keys',recog_key.keys)
        if recog_key.keys != None:  # we had a response
            trials_retrieval.addData('recog_key.rt', recog_key.rt)
            trials_retrieval.addData('recog_key.duration', recog_key.duration)
        # the Routine "recognition" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "source_memory" ---
        # create an object to store info about Routine source_memory
        source_memory = data.Routine(
            name='source_memory',
            components=[source_prompt, source_key],
        )
        source_memory.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from source_branching
        # If they pressed 'n' (No) during Recognition, skip this routine entirely
        if recog_key.keys == 'n':
            continueRoutine = False
        # create starting attributes for source_key
        source_key.keys = []
        source_key.rt = []
        _source_key_allKeys = []
        # store start times for source_memory
        source_memory.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        source_memory.tStart = globalClock.getTime(format='float')
        source_memory.status = STARTED
        thisExp.addData('source_memory.started', source_memory.tStart)
        source_memory.maxDuration = None
        # keep track of which components have finished
        source_memoryComponents = source_memory.components
        for thisComponent in source_memory.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "source_memory" ---
        source_memory.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_retrieval, 'status') and thisTrials_retrieval.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *source_prompt* updates
            
            # if source_prompt is starting this frame...
            if source_prompt.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                source_prompt.frameNStart = frameN  # exact frame index
                source_prompt.tStart = t  # local t and not account for scr refresh
                source_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(source_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'source_prompt.started')
                # update status
                source_prompt.status = STARTED
                source_prompt.setAutoDraw(True)
            
            # if source_prompt is active this frame...
            if source_prompt.status == STARTED:
                # update params
                pass
            
            # *source_key* updates
            waitOnFlip = False
            
            # if source_key is starting this frame...
            if source_key.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                source_key.frameNStart = frameN  # exact frame index
                source_key.tStart = t  # local t and not account for scr refresh
                source_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(source_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'source_key.started')
                # update status
                source_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(source_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(source_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if source_key.status == STARTED and not waitOnFlip:
                theseKeys = source_key.getKeys(keyList=['v','a'], ignoreKeys=["escape"], waitRelease=False)
                _source_key_allKeys.extend(theseKeys)
                if len(_source_key_allKeys):
                    source_key.keys = _source_key_allKeys[-1].name  # just the last key pressed
                    source_key.rt = _source_key_allKeys[-1].rt
                    source_key.duration = _source_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=source_memory,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                source_memory.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in source_memory.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "source_memory" ---
        for thisComponent in source_memory.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for source_memory
        source_memory.tStop = globalClock.getTime(format='float')
        source_memory.tStopRefresh = tThisFlipGlobal
        thisExp.addData('source_memory.stopped', source_memory.tStop)
        # check responses
        if source_key.keys in ['', [], None]:  # No response was made
            source_key.keys = None
        trials_retrieval.addData('source_key.keys',source_key.keys)
        if source_key.keys != None:  # we had a response
            trials_retrieval.addData('source_key.rt', source_key.rt)
            trials_retrieval.addData('source_key.duration', source_key.duration)
        # the Routine "source_memory" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "rem_know" ---
        # create an object to store info about Routine rem_know
        rem_know = data.Routine(
            name='rem_know',
            components=[rk_prompt, rk_response],
        )
        rem_know.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from rk_branching
        # If they pressed 'n' (No) during Recognition, skip this routine entirely
        if recog_key.keys == 'n':
            continueRoutine = False
        # create starting attributes for rk_response
        rk_response.keys = []
        rk_response.rt = []
        _rk_response_allKeys = []
        # store start times for rem_know
        rem_know.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        rem_know.tStart = globalClock.getTime(format='float')
        rem_know.status = STARTED
        thisExp.addData('rem_know.started', rem_know.tStart)
        rem_know.maxDuration = None
        # keep track of which components have finished
        rem_knowComponents = rem_know.components
        for thisComponent in rem_know.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "rem_know" ---
        rem_know.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_retrieval, 'status') and thisTrials_retrieval.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rk_prompt* updates
            
            # if rk_prompt is starting this frame...
            if rk_prompt.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                rk_prompt.frameNStart = frameN  # exact frame index
                rk_prompt.tStart = t  # local t and not account for scr refresh
                rk_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rk_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rk_prompt.started')
                # update status
                rk_prompt.status = STARTED
                rk_prompt.setAutoDraw(True)
            
            # if rk_prompt is active this frame...
            if rk_prompt.status == STARTED:
                # update params
                pass
            
            # *rk_response* updates
            waitOnFlip = False
            
            # if rk_response is starting this frame...
            if rk_response.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                rk_response.frameNStart = frameN  # exact frame index
                rk_response.tStart = t  # local t and not account for scr refresh
                rk_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rk_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rk_response.started')
                # update status
                rk_response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(rk_response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(rk_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if rk_response.status == STARTED and not waitOnFlip:
                theseKeys = rk_response.getKeys(keyList=['1','2','3'], ignoreKeys=["escape"], waitRelease=False)
                _rk_response_allKeys.extend(theseKeys)
                if len(_rk_response_allKeys):
                    rk_response.keys = _rk_response_allKeys[-1].name  # just the last key pressed
                    rk_response.rt = _rk_response_allKeys[-1].rt
                    rk_response.duration = _rk_response_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=rem_know,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                rem_know.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rem_know.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rem_know" ---
        for thisComponent in rem_know.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for rem_know
        rem_know.tStop = globalClock.getTime(format='float')
        rem_know.tStopRefresh = tThisFlipGlobal
        thisExp.addData('rem_know.stopped', rem_know.tStop)
        # check responses
        if rk_response.keys in ['', [], None]:  # No response was made
            rk_response.keys = None
        trials_retrieval.addData('rk_response.keys',rk_response.keys)
        if rk_response.keys != None:  # we had a response
            trials_retrieval.addData('rk_response.rt', rk_response.rt)
            trials_retrieval.addData('rk_response.duration', rk_response.duration)
        # the Routine "rem_know" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrials_retrieval as finished
        if hasattr(thisTrials_retrieval, 'status'):
            thisTrials_retrieval.status = FINISHED
        # if awaiting a pause, pause now
        if trials_retrieval.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_retrieval.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_retrieval'
    trials_retrieval.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[text_2, key_resp_2],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=end,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
