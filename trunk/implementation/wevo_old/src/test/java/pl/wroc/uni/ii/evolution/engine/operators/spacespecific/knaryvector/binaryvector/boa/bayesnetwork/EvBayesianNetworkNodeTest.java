package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.bayesnetwork;


import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.bayesnetwork.EvBayesianNetworkNode;
import junit.framework.TestCase;

public class EvBayesianNetworkNodeTest  extends TestCase{

  public void testNodeTest() {
    
    EvBayesianNetworkNode[] net = new EvBayesianNetworkNode[4];
    net[0] = new EvBayesianNetworkNode(0, 4);
    net[1] = new EvBayesianNetworkNode(1, 4);
    net[2] = new EvBayesianNetworkNode(2, 4);
    net[3] = new EvBayesianNetworkNode(3, 4);
    
    
    net[0].addChild(net[1]);
    net[1].addParent(net[0]);
    
    assertTrue(net[0].hasChild(net[1]));
    assertTrue(net[1].hasParent(net[0]));
    
    
    System.out.println(net[0]);
    
    
  }
}
