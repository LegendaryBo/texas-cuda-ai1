#use this scrip to automaticaly add required jars to maven repository
mvn install:install-file -DgroupId=eg4eem -DartifactId=eg4eem -Dversion=1.0.0 -Dpackaging=jar -Dfile=eg4eem.jar
mvn install:install-file -DgroupId=fdm -DartifactId=fdm -Dversion=1.1.0 -Dpackaging=jar -Dfile=fdm2.jar
mvn install:install-file -DgroupId=cos -DartifactId=cos -Dversion=05Nov2002 -Dpackaging=jar -Dfile=cos.jar
mvn install:install-file -DgroupId=Jama -DartifactId=Jama -Dversion=1.0.2 -Dpackaging=jar -Dfile=Jama-1.0.2.jar
mvn install:install-file -DgroupId=org.swig -DartifactId=swig -Dversion=1.3.31-1 -Dclassifier=x86-Linux-g++-static -Dpackaging=nar -Dfile=swig-1.3.31-1-i386-Linux-g++-static.nar
