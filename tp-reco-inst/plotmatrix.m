function plotmatrix(X, classes)
  % Repr?sentation d'un jeu de donn?es sous forme d'une matrice de nuages de points 2D
  dimensions = size(X,2);
  subplot(dimensions, dimensions, 1);
  n = max(classes);
  colors = ['r' 'g' 'b' 'm'];
  for i=1:dimensions,
      for j=i:dimensions,
          subplot(dimensions, dimensions,(i-1)*dimensions+j);
          for k=1:n,
              ii = find(classes == k);
              if i==j,
                x = -4:0.1:4;
                m = mean(X(ii,i));
                v = var(X(ii,i));
                y = exp(-(x-m).*(x-m)/(2*v))/sqrt(2*pi*v);
                hold on, plot(x,y, sprintf('%s',colors(k)));
              else,
                hold on, plot(X(ii,i),X(ii,j), sprintf('%s+',colors(k)));
              end;
          end;
      end;
   end;
%    subplot(dimensions, dimensions, dimensions+1);
%    for k=1:n,
%      hold on, plot(1,1, sprintf('%s+',colors(k)));
%    end;
   legend('Clarinette','Piano','Trompette','Violon');
%endfunction
