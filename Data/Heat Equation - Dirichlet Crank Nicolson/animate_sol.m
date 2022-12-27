function animate_sol(fig, x, u, u_min, u_max)
% size(x)
% size(u)
enable=1;

if(enable == 0)
    figure(fig);
    plot(x, u,'linewidth',1);
    title('Solution over time')
    xlabel('x')
    ylabel('u')
    hold off; %or hold on to show solution for every time step
    axis([0,max(x),u_min,u_max]);
    pause(0.20);
end

end