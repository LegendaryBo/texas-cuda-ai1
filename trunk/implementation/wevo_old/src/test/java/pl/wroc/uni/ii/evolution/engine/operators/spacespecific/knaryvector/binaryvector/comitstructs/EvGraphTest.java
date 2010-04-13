package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.comitstructs;

import java.lang.reflect.Field;

import junit.framework.TestCase;

/**
 * 
 * @author Kacper Gorski, Olgierd Humenczuk
 * 
 */

public class EvGraphTest extends TestCase {
 
  EvGraph test_graph = null;
  
  public void setUp() {
    int[][] m = new int[4][4];
    m[1][0] = 5;
    m[2][0] = 1;
    m[2][1] = 2;
    m[3][0] = 4;
    m[3][1] = 3;
    m[3][2] = 6;

    test_graph = new EvGraph(m);
  }
  
  public void tearDown() {
    test_graph = null;
  }
    
  private EvGraph getTestGraph() {
    EvGraph vertex1 = new EvGraph(0);
    EvGraph vertex2 = new EvGraph(1);
    EvGraph vertex3 = new EvGraph(2);
    EvGraph vertex4 = new EvGraph(3);
    EvGraph vertex5 = new EvGraph(4);
    EvGraph vertex6 = new EvGraph(5);

    Field nodes_number = null;
    try {
      nodes_number = vertex1.getClass().getDeclaredField( "nodes_number" );
      nodes_number.setAccessible( true );
      nodes_number.setInt( vertex1, 6 );
    } catch ( Exception e ) {
      fail( e.getMessage() );
    } 
    
    new EvEdge(7,vertex4, vertex5);
    new EvEdge(4,vertex4, vertex3);
    new EvEdge(2,vertex4, vertex1);
    new EvEdge(3,vertex1, vertex5);
    new EvEdge(10,vertex1, vertex2);
    new EvEdge(5,vertex5, vertex2);
    new EvEdge(9,vertex3, vertex2);
    new EvEdge(4,vertex3, vertex6);
    new EvEdge(7,vertex6, vertex2);
    return vertex1;
  }
  

  
  public void testgetMaximumSpanningTree() {

     EvTree tree = getTestGraph().getMaximumSpanningTree();
     
     assertEquals("node label:0\n" +
         "child: 1\n" +
         "--------------------\n" +
         "node label:1\n" +
         "child: 2\n" +
         "child: 4\n" +
         "child: 5\n" +
         "--------------------\n" +
         "node label:2\n" +
         "node label:4\n" +
         "child: 3\n" +
         "--------------------\n" +
         "node label:3\n" +
         "node label:5\n", tree.toString());
         
  }
  
  
}
