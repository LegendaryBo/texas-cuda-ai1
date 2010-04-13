package pl.wroc.uni.ii.evolution.solutionspaces;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.experimental.EvDecisionTreesSpace;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvAnswer;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvDecisionTreeIndividual;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvIsEven;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvIsDividedByFive;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvIsDividedByThree;

public class EvDecisionTreesSpaceTest extends TestCase {

  public void testBelongsTo() {
    EvDecisionTreesSpace<Integer> space = new EvDecisionTreesSpace<Integer>();
    
    space.addPossibleNode(new EvIsEven());
    space.addPossibleNode(new EvIsDividedByFive());
    EvDecisionTreeIndividual<Integer> tree = new EvDecisionTreeIndividual<Integer>(new EvIsEven());
    assertTrue(space.belongsTo(tree));
    tree.setNodeForAnswer(EvAnswer.YES, new EvDecisionTreeIndividual<Integer>(new EvIsDividedByFive()));
    assertTrue(space.belongsTo(tree));
    
    tree.setNodeForAnswer(EvAnswer.NO, new EvDecisionTreeIndividual<Integer>(new EvIsDividedByThree()));
    assertFalse(space.belongsTo(tree));
    tree.replace(tree, new EvDecisionTreeIndividual<Integer>(new EvIsDividedByThree()));
    assertFalse(space.belongsTo(tree));
    tree = tree.replace(tree, new EvDecisionTreeIndividual<Integer>(new EvIsDividedByFive()));
    
    
    assertTrue(space.belongsTo(new EvDecisionTreeIndividual<Integer>(new EvIsDividedByFive())));
    assertTrue(space.belongsTo(tree));
    
  }

}
