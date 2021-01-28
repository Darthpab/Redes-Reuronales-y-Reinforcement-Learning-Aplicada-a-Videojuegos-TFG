from py4j.java_gateway import get_field

class RuleBased(object):
  def __init__(self, gateway):
    self.gateway = gateway

  def close(self):
    pass

  def getInformation(self, frameData, isControl):
    self.frameData = frameData
    self.cc.setFrameData(self.frameData, self.player)
		
  def roundEnd(self, x, y, z):
    print(x)
    print(y)
    print(z)

  def getScreenData(self, sd):
    pass
  
  def initialize(self, gameData, player):
    self.inputKey = self.gateway.jvm.struct.Key()
    self.frameData = self.gateway.jvm.struct.FrameData()
    self.cc = self.gateway.jvm.aiinterface.CommandCenter()
    self.player = player
    self.gameData = gameData
    self.simulator = self.gameData.getSimulator()
    self.isGameJustStarted = True
    return 0

  def input(self):
    return self.inputKey

  def processing(self):
    if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
      self.isGameJustStarted = True
      return
  
    if not self.isGameJustStarted:
      self.frameData = self.simulator.simulate(self.frameData, self.player, None, None, 17)
    else:
      self.isGameJustStarted = False

    self.cc.setFrameData(self.frameData, self.player)
    distance = self.frameData.getDistanceX()
    
    my = self.frameData.getCharacter(self.player)
    energy = my.getEnergy()
    my_x = my.getX()
    my_state = my.getState()
    
    opp = self.frameData.getCharacter(not self.player)
    opp_x = opp.getX()
    opp_state = opp.getState()
    
    xDifference = my_x - opp_x
    
    if self.cc.getSkillFlag():
      self.inputKey = self.cc.getSkillKey()
      return
    self.inputKey.empty()
    self.cc.skillCancel()
    
    if (opp.getEnergy() >= 300) and (my.getHp()- opp.getHp() <= 300):

      self.cc.commandCall("FOR_JUMP _B B B")
    elif not my_state.equals(self.gateway.jvm.enumerate.State.AIR) and not my_state.equals(self.gateway.jvm.enumerate.State.DOWN):

      if distance > 150:

        self.cc.commandCall("FOR_JUMP")
      elif energy >= 300:

        self.cc.commandCall("STAND_D_DF_FC")
      elif (distance > 100) and (energy >= 50):

        self.cc.commandCall("STAND_D_DB_BB")
      elif opp_state.equals(self.gateway.jvm.enumerate.State.AIR): 

        self.cc.commandCall("STAND_F_D_DFA")
      elif distance > 100:
 
        self.cc.commandCall("6 6 6")
      else:

        self.cc.commandCall("B")
    elif ((distance <= 150) and (my_state.equals(self.gateway.jvm.enumerate.State.AIR) or my_state.equals(self.gateway.jvm.enumerate.State.DOWN))and
     (((self.gameData.getStageWidth() - my_x) >= 200) or (xDifference > 0)) and ((my_x >= 200) or xDifference < 0)):

      if energy >= 5:
        self.cc.commandCall("AIR_DB")
      else:
        self.cc.commandCall("B")
    else:
      self.cc.commandCall("B")

  class Java:
    implements = ["aiinterface.AIInterface"]
