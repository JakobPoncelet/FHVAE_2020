function List=l_expand(Val,Frq)
% l_expand: expand Val(ues) with Fr(e)q(uency) into List.
% List=l_expand(Val,Frq)
% inverse of packlist: repeat Val(i), Frq(i) times in List
% Val and Frq must have same length. length(List)=sum(Frq).
List=zeros(1,sum(Frq));
i=0;
for k=1:length(Frq),
 N=Frq(k);
 v=Val(k);
 List(i+1:i+N)=v(ones(1,N));
 i=i+N;
end
end
