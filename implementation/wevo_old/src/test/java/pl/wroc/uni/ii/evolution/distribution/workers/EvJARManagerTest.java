package pl.wroc.uni.ii.evolution.distribution.workers;

import java.io.FileInputStream;
import java.io.IOException;

import org.jmock.Mock;
import org.jmock.MockObjectTestCase;

import pl.wroc.uni.ii.evolution.distribution.workers.EvJARCacheImpl;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunication;

public class EvJARManagerTest extends MockObjectTestCase {

  public void testOne() throws IOException {
    
//    Mock manager_proxy = mock(EvManagmentServletCommunication.class);
//  
//    FileInputStream f_in = new FileInputStream(System.getProperty("user.dir") +
//        "/devel/tests/pl/wroc/uni/ii/evolution/distribution/workers/task.zip");
//    byte[] tmp = new byte[f_in.available()];
//    
//    f_in.read(tmp);
// 
//    manager_proxy.expects(once()).method("getJAR").with(eq(12L)).will(returnValue(tmp));
//    manager_proxy.expects(once()).method("getJAR").with(eq(15L)).will(returnValue(tmp));
//    
//    
//    EvJARCacheImpl jar_manager = new EvJARCacheImpl((EvManagmentServletCommunication) manager_proxy.proxy());
//    
//    jar_manager.init(1);
//    
//    jar_manager.getJARUrl(12, 1);
//    jar_manager.getJARUrl(13, 1);
//    jar_manager.getJARUrl(13, 1);
//    jar_manager.getJARUrl(15, 2);
//    
//    jar_manager.dispose();
//    
  }

}
