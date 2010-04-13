package pl.wroc.uni.ii.evolution.experimental.decisiontree;

import org.jmock.Mock;
import org.jmock.MockObjectTestCase;

import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvAnswer;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvDecision;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvDecisionTreeIndividual;
import pl.wroc.uni.ii.evolution.utils.EvIRandomizer;

public class EvDecisionTreeTest extends MockObjectTestCase {

  /**
   * Test that DecisionTree node delegets making a decision
   * to DecisionStrategy. It makes sure that the answer returned
   * is the same that Strategy gave no matter if it was YES, NO, or
   * DUNNO.
   * 
   */
  public void testLeafCreation() throws Exception {
    
    for (EvAnswer answer : EvAnswer.values()) {
      test_node_with_mock_that_returns(answer);
    }
  }

  private void test_node_with_mock_that_returns(EvAnswer answer) {
    EvDecision<Integer> mock_of_decision_strategy = 
      create_mock_expecting_to_be_called_decide_and_returning(answer);
    EvDecisionTreeIndividual<Integer> decision_tree = 
      new EvDecisionTreeIndividual<Integer>( mock_of_decision_strategy );
    
    assertEquals(answer, decision_tree.decide(new Integer(1)));
  }

  @SuppressWarnings("unchecked")
  private EvDecision<Integer> create_mock_expecting_to_be_called_decide_and_returning(EvAnswer answer) {
    Mock mock_settings = mock(EvDecision.class);
    mock_settings.expects(once()).method("decide").will(returnValue(answer));
    EvDecision<Integer> mock = (EvDecision<Integer>) mock_settings.proxy();
    return mock;
  }
  
  
  /** Test if node delegates making a further decision 
   * to an appropriate child node (in case he has one).
   */
  public void testSetSubtree() throws Exception {
    EvDecision<Integer> yes_mock = create_mock_expecting_to_be_called_decide_and_returning(EvAnswer.YES);
    EvDecisionTreeIndividual<Integer> parent = 
      new EvDecisionTreeIndividual<Integer>(yes_mock);
    
    EvAnswer answer = EvAnswer.NO;
    EvDecision<Integer> no_mock = create_mock_expecting_to_be_called_decide_and_returning(answer);
    EvDecisionTreeIndividual<Integer> child = 
      new EvDecisionTreeIndividual<Integer>(no_mock);
    
    parent.setNodeForAnswer(EvAnswer.YES,child);
    
    assertEquals( answer, parent.decide(new Integer(1)) );
  }
  
  public void testAddingNodeTwice() {
    EvDecision<Integer> yes_mock =
      create_mock_expecting_to_be_called_decide_and_returning(EvAnswer.YES);
    EvDecisionTreeIndividual<Integer> parent = 
      new EvDecisionTreeIndividual<Integer>(yes_mock);
    
    //child1.decide should not be called
    EvDecisionTreeIndividual<Integer> child1 = create_fake();
    
    EvDecision<Integer> no_mock =
      create_mock_expecting_to_be_called_decide_and_returning(EvAnswer.NO);
    EvDecisionTreeIndividual<Integer> child2 = 
      new EvDecisionTreeIndividual<Integer>(no_mock);
    
    parent.setNodeForAnswer(EvAnswer.YES,child1);
    //child2 should take place of child1
    parent.setNodeForAnswer(EvAnswer.YES,child2);
    
    assertEquals(EvAnswer.NO,parent.decide(new Integer(0)));
  }

  public static EvDecisionTreeIndividual<Integer> create_fake() {
    EvDecisionTreeIndividual<Integer> child1 = 
      new EvDecisionTreeIndividual<Integer>(new EvDecision<Integer>() {

        public EvAnswer decide(Integer arg) {
          throw new RuntimeException("should not be called");
        }
        @Override
        public boolean equals(Object other) {
          return this.getClass().equals(other.getClass());
        }
      });
    return child1;
  }

  
  /**
   * Ensure that replace will just clone should there be given
   * a node that is not a descendant.
   * <p>
   * Ensure that it is actualy a clone.
   * 
   * @throws Exception
   */
  public void testRepalcement_wrongNode() throws Exception {
    EvDecisionTreeIndividual<Integer> node = create_fake();
    node.setNodeForAnswer(EvAnswer.NO, create_fake());
    
    EvDecisionTreeIndividual<Integer> result =
      node.replace(create_fake(),create_fake());
    assertEquals(node,result);
    assertNotSame(node,result);
  }
  
  /**
   * Replace the root itself.
   * <p>
   * Ensure that the result of replacing root is the very object
   * given as substitute.
   * 
   * @throws Exception
   */
  public void testRepalcement_oneNode() throws Exception {
    EvDecisionTreeIndividual<Integer> expected = create_fake();
    
    EvDecisionTreeIndividual<Integer> node = create_fake();
    node.setNodeForAnswer(EvAnswer.NO, create_fake());
    
    EvDecisionTreeIndividual<Integer> result =
      node.replace(node,expected);
    
    assertSame(expected,result);
  }
  
  /**
   * Test replacing child.
   * 
   * @throws Exception
   */
  public void testRepalcement_child() throws Exception {
    EvDecisionTreeIndividual<Integer> node = create_fake();
    EvDecisionTreeIndividual<Integer> child = create_fake();
    node.setNodeForAnswer(EvAnswer.NO,child);
    child.setNodeForAnswer(EvAnswer.YES, create_fake());
    
    EvDecisionTreeIndividual<Integer> substitute = create_fake();
    EvDecisionTreeIndividual<Integer> expected = create_fake();
    expected.setNodeForAnswer(EvAnswer.NO, substitute);
    
    EvDecisionTreeIndividual<Integer> result =
      node.replace(child,substitute);
    
    assertEquals(expected,result);
  }
  // ===========================
  // Tests for random descendant
  
  // two nodes in the tree, expect a child
  public void testGetRandomDescendant_getChild() throws Exception {
    EvDecisionTreeIndividual<Integer> node = create_fake();
    EvDecisionTreeIndividual<Integer> child = create_fake();
    node.setNodeForAnswer(EvAnswer.NO,child);
    
    Mock random = mock(EvIRandomizer.class);
    random.expects(once()).method("nextInt").with(eq(2)).will(returnValue(1));
    
    EvDecisionTreeIndividual<Integer> result =
      node.randomDescendant((EvIRandomizer)random.proxy());
    
    assertEquals(child,result);
  }
  
  // two nodes in the tree, expect parent
  public void testGetRandomDescendant_getParent() throws Exception {
    EvDecisionTreeIndividual<Integer> node = create_fake();
    EvDecisionTreeIndividual<Integer> child = create_fake();
    node.setNodeForAnswer(EvAnswer.NO,child);
    
    Mock random = mock(EvIRandomizer.class);
    random.expects(once()).method("nextInt").with(eq(2)).will(returnValue(0));
    
    EvDecisionTreeIndividual<Integer> result =
      node.randomDescendant((EvIRandomizer)random.proxy());
    
    assertEquals(node,result);
  }
  
  // one node in the tree, expect that very node
  public void testGetRandomDescendant_oneNodeTree() throws Exception {
    EvDecisionTreeIndividual<Integer> node = create_fake();
    
    Mock random = mock(EvIRandomizer.class);
    random.expects(once()).method("nextInt").with(eq(1)).will(returnValue(0));
    
    EvDecisionTreeIndividual<Integer> result =
      node.randomDescendant((EvIRandomizer)random.proxy());
    
    assertEquals(node,result);
  }
}
